"""Binary Spray and Wait helpers for encounter-based DTN simulations."""
import string

from encounter import EncounterManager
from pywisim import EventLoop, Node, WirelessNetwork


class BinarySprayWaitNode(Node):
    def __init__(self, nid, tracker=None):
        super().__init__(nid)
        self.bundles = {}                         # mid -> {mid, src, dst, created_at, copies}
        self.tracker = tracker or {}

    def inject(self, mid, dest, copies):
        if copies < 1:
            raise ValueError("copies must be >= 1")
        self.bundles[mid] = {
            'mid': mid,
            'src': self.nid,
            'dst': dest,
            'created_at': self.net.loop.time,
            'copies': copies,
        }
        self.net.log(f"{self.nid}: originated '{mid}' for {dest} with {copies} copies")

    def on_receive(self, msg, sender):
        kind = msg[0]
        if kind == 'ENCOUNTER':
            self.unicast(sender, ('SUMMARY', tuple(sorted(self.bundles))))
        elif kind == 'SUMMARY':
            self._handle_summary(sender, set(msg[1]))
        elif kind == 'SPRAY':
            self._handle_spray(sender, msg[1], msg[2])
        elif kind == 'DELIVER':
            self._handle_deliver(sender, msg[1])

    def _handle_summary(self, sender, peer_bundle_ids):
        for mid, record in sorted(self.bundles.items()):
            if mid in peer_bundle_ids:
                continue
            bundle = self._bundle_meta(record)
            if record['dst'] == sender:
                self._count_data_tx()
                self.net.log(f"{self.nid}: delivering '{mid}' directly to {sender}")
                self.unicast(sender, ('DELIVER', bundle))
            elif record['copies'] > 1:
                share = record['copies'] // 2
                if share:
                    record['copies'] -= share
                    self._count_data_tx()
                    self.net.log(
                        f"{self.nid}: sprayed {share} copies of '{mid}' to {sender}; "
                        f"kept {record['copies']}"
                    )
                    self.unicast(sender, ('SPRAY', bundle, share))

    def _handle_spray(self, sender, bundle, share):
        mid = bundle['mid']
        if mid in self.bundles:
            return
        self.bundles[mid] = {**bundle, 'copies': share}
        self.net.log(f"  {self.nid} got {share} copies of '{mid}' from {sender}")

    def _handle_deliver(self, sender, bundle):
        mid = bundle['mid']
        if mid not in self.bundles:
            self.bundles[mid] = {**bundle, 'copies': 1}
        if bundle['dst'] != self.nid or self.tracker.get('delivered_at') is not None:
            return
        self.tracker['delivered_at'] = self.net.loop.time
        self.tracker['delivered_by'] = sender
        self.tracker['total_tx_at_delivery'] = self.net._mid
        self.tracker['data_tx_at_delivery'] = self.tracker.get('data_tx', 0)
        self.net.log(f"  {self.nid} delivered '{mid}' from {sender}")
        stop = self.tracker.get('stop')
        if stop:
            stop()

    def _count_data_tx(self):
        self.tracker['data_tx'] = self.tracker.get('data_tx', 0) + 1

    @staticmethod
    def _bundle_meta(record):
        return {k: record[k] for k in ('mid', 'src', 'dst', 'created_at')}


def default_node_ids(count):
    if count <= 26:
        return list(string.ascii_uppercase[:count])
    return [f'N{i}' for i in range(count)]


def run_binary_spray_wait(
    node_count=8,
    copies=4,
    rate=1.5,
    duration=1.0,
    seed=42,
    until=30.0,
    node_ids=None,
    source=None,
    dest=None,
    verbose=False,
):
    if node_ids is None:
        node_ids = default_node_ids(node_count)
    else:
        node_ids = list(node_ids)
        node_count = len(node_ids)
    if node_count < 2:
        raise ValueError("Binary Spray and Wait requires at least two nodes")

    source = source or node_ids[0]
    dest = dest or node_ids[-1]
    if source == dest:
        raise ValueError("source and destination must be different")

    loop = EventLoop()
    net = WirelessNetwork(loop, loss=0.0, tx_time=0.01, verbose=verbose, seed=seed)
    tracker = {
        'data_tx': 0,
        'delivered_at': None,
        'delivered_by': None,
        'total_tx_at_delivery': None,
        'data_tx_at_delivery': None,
    }

    for nid in node_ids:
        net.add_node(BinarySprayWaitNode(nid, tracker), 0, 0)

    enc = EncounterManager(net, rate=rate, duration=duration)
    tracker['stop'] = enc.stop

    bundle_id = f'{source}->{dest}'
    start_time = 0.1
    loop.schedule(start_time, net.nodes[source].inject, bundle_id, dest, copies)
    enc.start()
    loop.run(until=until)

    delivered = tracker['delivered_at'] is not None
    delay = None if not delivered else tracker['delivered_at'] - start_time
    total_tx = tracker['total_tx_at_delivery'] if delivered else net._mid
    data_tx = tracker['data_tx_at_delivery'] if delivered else tracker['data_tx']
    holders = {
        nid: net.nodes[nid].bundles[bundle_id]['copies']
        for nid in node_ids
        if bundle_id in net.nodes[nid].bundles
    }

    return {
        'bundle_id': bundle_id,
        'delivered': delivered,
        'delay': delay,
        'total_tx': total_tx,
        'data_tx': data_tx,
        'delivered_by': tracker['delivered_by'],
        'delivery_time': tracker['delivered_at'],
        'source': source,
        'dest': dest,
        'holders': holders,
        'node_ids': node_ids,
        'copies': copies,
        'node_count': node_count,
        'seed': seed,
        'until': until,
        'final_time': loop.time,
    }
