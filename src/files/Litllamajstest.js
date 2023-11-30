function fibonacci(num) {
	let num1 = 0;
	let num2 = 1;
	let sum;
	let i = 0;
	for (i = 0; i < num; i++) {
		sum = num1 + num2;
		num1 = num2;
		num2 = sum;
	}
	return num2;
}

console.log("Fibonacci(5): " + fibonacci(5));
console.log("Fibonacci(8): " + fibonacci(8));
