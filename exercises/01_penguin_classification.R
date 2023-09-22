# Load necessary libraries
library(palmerpenguins)
library(torch)
library(torchvision)
library(data.table)
library(ggplot2)
library(DataLoader)  # Load the dataloader package


# Set random seed for reproducibility
set.seed(42)

# Load the penguins dataset
data <- penguins

# Display the dataset and explore its features
print(head(data))
print(names(data))

# Select features and target variables
input_keys <- c("bill_length_mm", "body_mass_g")
target_keys <- c("species")

# Create a custom dataset class
PenguinDataset <- function(data, input_keys, target_keys, train) {
    dataset <- data.table(data)
    dataset <- dataset[, c(input_keys, target_keys), with = FALSE]
    valid_data <- NULL  # Define valid_data outside of the conditional
    if (train) {
        # Split the data into training and validation sets (e.g., 80% train, 20% validation)
        split_index <- round(0.8 * nrow(dataset))
        train_data <- dataset[1:split_index, ]
        valid_data <- dataset[(split_index + 1):nrow(dataset), ]
        data <- train_data
    } else {
        data <- valid_data
    }
    # Define transformation functions for inputs and targets if needed
    x_tfms <- list()
    y_tfms <- list()
    obj <- list(
        data = data,
        x_tfms = x_tfms,
        y_tfms = y_tfms
    )
    class(obj) <- "PenguinDataset"
    return(obj)
}

# Instantiate the PenguinDataset for training and validation
train_dataset <- PenguinDataset(data, input_keys, target_keys, train = TRUE)
valid_dataset <- PenguinDataset(data, input_keys, target_keys, train = FALSE)

# Define data loaders
batch_size <- 32
train_loader <- dataloader(train_dataset, batch_size = batch_size,
    shuffle = TRUE, drop_last = TRUE)
valid_loader <- DataLoader(valid_dataset, batch_size = batch_size, shuffle = FALSE)

# Define the neural network architecture
FCNet <- nn_module(
  "FCNet",

  initialize = function(input_size, hidden_size, output_size, dropout_prob) {
    self$input_size <- input_size
    self$hidden_size <- hidden_size
    self$output_size <- output_size
    self$dropout_prob <- dropout_prob

    self$fc1 <- nn_linear(input_size, hidden_size)
    self$relu <- nn_relu()
    self$dropout <- nn_dropout(dropout_prob)
    self$fc2 <- nn_linear(hidden_size, output_size)
    self$softmax <- nn_softmax()
  },

  forward = function(x) {
    x <- self$fc1(x)
    x <- self$relu(x)
    x <- self$dropout(x)
    x <- self$fc2(x)
    x <- self$softmax(x)
    return(x)
  }
)

# Instantiate the model
input_size <- length(input_keys)
hidden_size <- 64
output_size <- length(unique(data[, target_keys]))
dropout_prob <- 0.5
model <- FCNet(input_size, hidden_size, output_size, dropout_prob)

# Define loss function and optimizer
loss_fn <- nn_bce_loss()
optimizer <- optim_adam(model$parameters(), lr = 0.001)

# Training and validation functions
train_one_epoch <- function(model, train_loader, loss_fn, optimizer) {
  model$train()
  train_loss <- 0.0
  total_correct <- 0
  total_samples <- 0

  for (batch in train_loader) {
    inputs <- batch$data$x
    targets <- batch$data$y

    optimizer$zero_grad()

    outputs <- model(inputs)
    loss <- loss_fn(outputs, targets)

    loss$backward()
    optimizer$step()

    train_loss <- train_loss + loss$item()
    total_samples <- total_samples + nrow(inputs)
  }

  avg_loss <- train_loss / length(train_loader)

  return(list(avg_loss = avg_loss))
}

validate_one_epoch <- function(model, valid_loader, loss_fn) {
  model$eval()
  valid_loss <- 0.0
  total_correct <- 0
  total_samples <- 0

  for (batch in valid_loader) {
    inputs <- batch$data$x
    targets <- batch$data$y

    with(torch.no_grad(), {
      outputs <- model(inputs)
    })

    loss <- loss_fn(outputs, targets)
    valid_loss <- valid_loss + loss$item()
    total_samples <- total_samples + nrow(inputs)
  }

  avg_loss <- valid_loss / length(valid_loader)

  return(list(avg_loss = avg_loss))
}

# Training loop
num_epochs <- 10
train_losses <- vector("numeric", num_epochs)
valid_losses <- vector("numeric", num_epochs)

for (epoch in 1:num_epochs) {
  # Training
  train_metrics <- train_one_epoch(model, train_loader, loss_fn, optimizer)
  train_losses[epoch] <- train_metrics$avg_loss

  # Validation
  valid_metrics <- validate_one_epoch(model, valid_loader, loss_fn)
  valid_losses[epoch] <- valid_metrics$avg_loss

  cat("Epoch: ", epoch, " | Train Loss: ", train_losses[epoch], " | Validation Loss: ", valid_losses[epoch], "\n")
}

# Plot training and validation losses
epochs <- 1:num_epochs
loss_data <- data.frame(Epoch = rep(epochs, 2),
                        Loss = c(train_losses, valid_losses),
                        Type = rep(c("Train", "Validation"), each = num_epochs))

ggplot(loss_data, aes(x = Epoch, y = Loss, color = Type)) +
  geom_line() +
  labs(title = "Training and Validation Loss", x = "Epoch", y = "Loss") +
  scale_color_manual(values = c("Train" = "blue", "Validation" = "red"))

# Visualize some results
# Construct a tensor of new inputs to run the model over
new_inputs <- as_tensor(data.table(bill_length_mm = c(40.0, 30.0, 45.0),
                                   body_mass_g = c(3500.0, 2000.0, 5000.0)))

# Place model in eval mode and run over inputs with no_grad
model$eval()
with(torch.no_grad(), {
  outputs <- model(new_inputs)
})

# Transform the raw output back to human-readable format
predicted_classes <- argmax(outputs, dim = 2)
predicted_species <- unique(data[, target_keys])[predicted_classes]
print(predicted_species)
