import flwr as fl
import numpy as np
from strategy import Strategy
from client import FlowerClient
from pathlib import Path
from task import get_partitions_and_label
from flwr.common import Context


partitions, label = get_partitions_and_label()

def client_fn(context: Context):
    # Print the context object to understand its structure
    print(f"Context: {context}")
    
    # Extract client ID from node_config
    cid = context.node_config.get("partition-id")
    
    if cid is None:
        raise AttributeError("Context object does not have 'partition-id' in node_config.")
    
    return FlowerClient(cid, partitions[int(cid)]).to_client()

# Start Flower server
hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=fl.server.ServerConfig(num_rounds=1000),
    strategy=Strategy(label),
)

#results_dir = Path("_static/results")
#results_dir.mkdir(exist_ok=True)
#np.save(str(results_dir / "hist.npy"), hist)
