"""Binary Spray and Wait over an intermittently connected mobile network (DTN)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spray_wait import run_binary_spray_wait


result = run_binary_spray_wait(
    node_ids=list('ABCDEFGH'),
    source='A',
    dest='H',
    copies=4,
    rate=1.5,
    duration=1.0,
    seed=42,
    until=30.0,
    verbose=True,
)

print("\nSummary:")
print(f"  Delivered: {result['delivered']}")
print(f"  Delay: {result['delay']:.2f}" if result['delay'] is not None else "  Delay: not delivered")
print(f"  Total transmissions: {result['total_tx']}")
print(f"  Data transmissions: {result['data_tx']}")
print(f"  Delivered by: {result['delivered_by']}")
print(f"  Holders at stop: {result['holders']}")
