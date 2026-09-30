import torch
import matplotlib.pyplot as plt

def get_device(prefer_mps=False):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and prefer_mps:
        return "mps"
    return "cpu"

def get_memory_consumption(device):
    if device == "mps":
        return torch.mps.current_allocated_memory()/1e9
    elif device == "cuda":
        return torch.cuda.memory_allocated(0)/1e9

def print_memory_consumption(device):
    if device == "mps":
        allocated_ram = torch.mps.current_allocated_memory()
        print(f"Allocated RAM: {allocated_ram/1e9:.2f} GB")
    elif device == "cuda":
        allocated_ram = torch.cuda.memory_allocated(0)
        max_allocated_ram = torch.cuda.max_memory_allocated(0)
        print(f"Current allocated RAM: {allocated_ram/1e9:.2f} GB, max: {max_allocated_ram/1e9:.2f} GB")

def plot_images(images, n_subplots):
    fig, axes = plt.subplots(1, min(len(images),n_subplots), figsize=(15, 10))

    for i, ax in enumerate(axes.flat):
        img = images[i]
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def torch_wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    # Ensure that the input tensors are batched
    assert u_values.dim() == 2 and v_values.dim() == 2, "Input tensors must be 2-dimensional (batch_size, num_values)"

    batch_size, u_size = u_values.shape
    _, v_size = v_values.shape

    # Sort the values
    u_sorter = torch.argsort(u_values, dim=1)
    v_sorter = torch.argsort(v_values, dim=1)

    # Concatenate and sort all values for each batch
    all_values = torch.cat((u_values, v_values), dim=1)
    all_values, _ = torch.sort(all_values, dim=1)
    # Compute differences between successive values
    deltas = torch.diff(all_values, dim=1)

    # Get the respective positions of the values of u and v among the values of both distributions
    all_continue = all_values[:, :-1].contiguous()
    u_cdf_indices = torch.searchsorted(u_values.gather(1, u_sorter).contiguous(), all_continue, right=True)
    v_cdf_indices = torch.searchsorted(v_values.gather(1, v_sorter).contiguous(), all_continue, right=True)

    # Calculate the CDFs of u and v using their weights, if specified
    if u_weights is None:
        u_cdf = u_cdf_indices.float() / u_size
    else:
        u_sorted_cumweights = torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(u_weights.gather(1, u_sorter), dim=1)), dim=1)
        u_cdf = u_sorted_cumweights.gather(1, u_cdf_indices) / u_sorted_cumweights[:, -1].unsqueeze(1)

    if v_weights is None:
        v_cdf = v_cdf_indices.float() / v_size
    else:
        v_sorted_cumweights = torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(v_weights.gather(1, v_sorter), dim=1)), dim=1)
        v_cdf = v_sorted_cumweights.gather(1, v_cdf_indices) / v_sorted_cumweights[:, -1].unsqueeze(1)

    return torch.sum(torch.abs(u_cdf - v_cdf) * deltas, dim=1)

def find_longest_repeated_block(numbers: list[int]):
        if not numbers:
            return 0, -1, -1  # Handle empty list case

        max_length = 0
        current_length = 0
        max_start_index = 0
        max_end_index = -1
        current_start_index = 0

        for i in range(len(numbers)):
            if i > 0 and numbers[i] == numbers[i-1]:
                current_length += 1
            else:
                # Check if the previous block was the longest
                if current_length > max_length:
                    max_length = current_length
                    max_start_index = current_start_index
                    max_end_index = i - 1

                # Reset for the new block
                current_length = 1
                current_start_index = i

        # Final check for the last block
        if current_length > max_length:
            max_length = current_length
            max_start_index = current_start_index
            max_end_index = len(numbers) - 1

        return max_start_index, max_end_index, numbers[max_start_index]
