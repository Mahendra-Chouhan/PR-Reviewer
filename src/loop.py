# More Complex Python Program

# Function to add two numbers
def add_numbers(a, b):
    return a + b

# Function to subtract two numbers
def subtract_numbers(a, b):
    return a - b

# Number pairs input by the user
num_pairs = int(input("Enter the number of pairs you want to calculate: "))

# List to store results
results = []

# Loop to get user input and perform calculations
for _ in range(num_pairs):
    num1 = float(input("Enter the first number: "))
    num2 = float(input("Enter the second number: "))

    # Perform addition and subtraction
    sum_result = add_numbers(num1, num2)
    difference_result = subtract_numbers(num1, num2)

    # Store results in a dictionary
    result_entry = {
        "Numbers": (num1, num2),
        "Sum": sum_result,
        "Difference": difference_result
    }

    # Append the result to the list
    results.append(result_entry)

# Print results
print("\nResults:")
for result in results:
    print(f"Numbers: {result['Numbers']}, Sum: {result['Sum']}, Difference: {result['Difference']}")
