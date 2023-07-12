# Contributing to paradigm-client

## Testing the code

Testing is an essential part of the development process, ensuring the reliability and correctness of the codebase. Our repository includes a **tests** folder that contains different tests to validate the functionality of the code. The test folder follows the same architecture as the source code to be tested.

To run the tests, we utilize the **pytest** module, a popular testing framework for Python. Here's how you can test the code:

1. Ensure that you have **pytest** installed on your local machine. If not, you can install it using the following command:
   ```
   pip install pytest
   ```

2. Open your terminal or command prompt.
3. Navigate to the root directory of the cloned repository.

4. To run all the tests, use the following command:
   ```
   pytest
   ```

   This command will automatically discover and execute all the tests within the **tests** folder.

5. If you want to run specific tests or tests within a specific directory, you can provide the path to the desired test file or directory. For example:
   ```
   pytest tests/test_file.py
   ```

   Replace **test_file.py** with the name of the specific test file you want to run.

6. Observe the test results in the terminal. The output will indicate whether each test passed or failed, along with any error messages or stack traces.

7. If any tests fail, you can investigate the issue by examining the error messages and stack traces. Make the necessary adjustments to the code to fix the failing tests.

8. After making changes, repeat the testing process to ensure that all tests pass successfully.

Remember to include any new tests for the features or bug fixes you contribute to the repository. This helps maintain the integrity of the codebase and ensures that future modifications do not introduce regressions.

Happy testing!