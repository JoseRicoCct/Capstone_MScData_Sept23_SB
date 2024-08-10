import fedml
from fedml import FedMLRunner
import logging

if __name__ == "__main__":
    args = fedml.init()

    # Initialize device
    device = fedml.device.get_device(args)

    # Load data
    dataset, output_dim = fedml.data.load(args)

    # Load model
    model = fedml.model.create(args, output_dim)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Lists to store accuracy and loss per round
    accuracy_summary = []
    loss_summary = []

    # Start the federated training
    fedml_runner = FedMLRunner(args, device, dataset, model)

    # Run the training
    for round_idx in range(args.comm_round):
        fedml_runner.run()

        # Retrieve metrics after the round
        # Replace with actual methods to get accuracy and loss
        # Example: accuracy, loss = fedml_runner.get_metrics()
        # Simulated retrieval for demonstration:
        metric_result_in_current_round = (0.01 + round_idx * 0.01, 1 - round_idx * 0.01)  # Simulated
        accuracy, loss = metric_result_in_current_round

        # Store accuracy and loss for the round
        accuracy_summary.append(accuracy)
        loss_summary.append(loss)

    # Print the summary of accuracies and losses at the end of all rounds
    logging.info("\n===== Training Summary =====")
    for i in range(args.comm_round):
        logging.info(f"Round {i + 1}:Accuracy = {accuracy_summary[i]:.4f}, Loss = {loss_summary[i]:.4f}")
    logging.info("===========================\n")
