## FunctionDef main(game_class, agent_class, ckpt_filepath, num_simulations)
# Documentation for `UserAuthenticationService`

## Overview

The `UserAuthenticationService` is a critical component of our application designed to handle user authentication processes securely and efficiently. This service provides methods for user login, logout, registration, and token management.

## Class Hierarchy

- **BaseClass**: Service
  - **UserAuthenticationService**

## Properties

### `private string _apiKey`

**Description**: A private key used for securing API communications between services.

**Usage**: Not directly accessible or modifiable by external classes. Used internally to authenticate service-to-service requests.

### `private ITokenManager _tokenManager`

**Description**: An instance of the `ITokenManager` interface responsible for generating and managing authentication tokens.

**Usage**: The `_tokenManager` is used to create, validate, and revoke access tokens during user sessions.

## Methods

### `public UserRegistrationResult RegisterUser(string username, string password)`

**Description**: Registers a new user with the provided credentials.

**Parameters**:
- **username** (string): The unique username for the new user.
- **password** (string): The password associated with the username.

**Returns**:
- **UserRegistrationResult**: An object containing the registration status and any relevant messages or errors.

### `public UserLoginResult LoginUser(string username, string password)`

**Description**: Authenticates a user based on their credentials.

**Parameters**:
- **username** (string): The username of the user attempting to log in.
- **password** (string): The password used for authentication.

**Returns**:
- **UserLoginResult**: An object containing the login status and any relevant messages or errors, including an access token if successful.

### `public void LogoutUser(string userId)`

**Description**: Logs out a user by invalidating their current session.

**Parameters**:
- **userId** (string): The unique identifier of the user to be logged out.

**Returns**: None

### `public bool ValidateToken(string token)`

**Description**: Validates an access token to ensure it is valid and not expired.

**Parameters**:
- **token** (string): The access token to validate.

**Returns**:
- **bool**: True if the token is valid, otherwise false.

## Exceptions

- **ArgumentException**: Thrown when required parameters are missing or invalid.
- **UnauthorizedAccessException**: Thrown when the provided credentials do not match any registered user.

## Example Usage

### Registration and Login

```csharp
UserRegistrationResult registrationResult = userService.RegisterUser("john_doe", "securepassword123");
if (registrationResult.Success)
{
    UserLoginResult loginResult = userService.LoginUser("john_doe", "securepassword123");
    if (loginResult.Success)
    {
        string accessToken = loginResult.AccessToken;
        // Use the access token for API requests
    }
}
```

### Logout

```csharp
userService.LogoutUser("userId123456");
```

## Notes

- The `UserAuthenticationService` is designed to be stateless, ensuring that it can handle concurrent user sessions without issues.
- All communication with the service should use HTTPS to ensure data security.

For more detailed information and advanced usage scenarios, please refer to the application's API documentation or contact the development team.
### FunctionDef node_to_str(node_i, reward, discount)
**node_to_str**: The function of `node_to_str` is to convert a node into a string representation including its reward, discount factor, value, and visits.

**parameters**:
· parameter1: `node_i`: An integer representing the index of the node.
· parameter2: `reward`: A float representing the reward associated with the node. Default value is 0.
· parameter3: `discount`: A float representing the discount factor used in calculating future rewards. Default value is 1.

**Code Description**: The function `node_to_str` generates a string representation of a tree node, providing details such as the node index (`node_i`), reward, discount factor, value, and visits. This function is useful for debugging or visualizing nodes within a search tree structure in a human-readable format.

The function constructs a multi-line string using formatted string literals (f-strings) to include:
1. The node index: `f"{node_i}\n"`
2. The reward with two decimal places: `f"Reward: {reward:.2f}\n"`
3. The discount factor with two decimal places: `f"Discount: {discount:.2f}\n"`
4. The value of the node, which is obtained from the `tree.node_values` array at position `[batch_index, node_i]`, formatted to two decimal places: `f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"`
5. The number of visits to the node, retrieved from the `tree.node_visits` array at position `[batch_index, node_i]`: `f"Visits: {tree.node_visits[batch_index, node_i]}\n"`

**Note**: Ensure that `batch_index`, `node_values`, and `node_visits` are correctly defined in the surrounding context or passed as arguments. If they are not provided, the function will use default values for reward (0) and discount (1).

**Output Example**: A possible return value could be:
```
2
Reward: 5.34
Discount: 0.98
Value: 7.65
Visits: 10
```
***
### FunctionDef edge_to_str(node_i, a_i)
**edge_to_str**: The function of `edge_to_str` is to convert an edge (or node) from the search tree into a formatted string representation.

**parameters**: 
· parameter1: `node_i`: An integer representing the index of the current node.
· parameter2: `a_i`: An integer representing the action or child node associated with the current node.

**Code Description**: The function `edge_to_str` is designed to generate a human-readable string representation for each edge in a search tree, typically used in debugging or visualization purposes. Here’s a detailed breakdown:

1. **Node Index Calculation**: 
   ```python
   node_index = jnp.full([batch_size], node_i)
   ```
   This line creates an array of size `batch_size` filled with the value `node_i`, which represents the index of the current node in the tree.

2. **Probability Calculation**:
   ```python
   probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
   ```
   The function uses JAX's `softmax` to convert the logits (raw scores) for the children of the current node into probabilities. This is done using the slice `tree.children_prior_logits[batch_index, node_i]`, which retrieves the relevant logits for the specified batch index and node index.

3. **String Formatting**:
   ```python
   return (
       f"a{a_i}\n"
       f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"
       f"p: {probs[a_i]:.2f}\n"
   )
   ```
   The function returns a formatted string that includes:
   - `a{a_i}`: A label indicating the action or child node.
   - `Q: <value>`: The Q-value (quality value) for the specified action, rounded to two decimal places.
   - `p: <value>`: The probability of taking the specified action, also rounded to two decimal places.

The function ensures that each edge in the search tree is represented with its associated action and relevant values, providing a clear and concise view of the node’s state and actions.

**Note**: Ensure that the batch index (`batch_index`) is correctly defined and accessible within the scope where `edge_to_str` is called. Also, make sure that `tree.qvalues()` and `tree.children_prior_logits` are appropriately defined in your codebase to avoid runtime errors.

**Output Example**: An example output could be:
```
a0
Q: 0.95
p: 0.23
```
***
