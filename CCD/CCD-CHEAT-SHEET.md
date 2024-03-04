1. Remove dead code - commented out code
   ```
    # This is dead code
   ------------------------
  
    # def display():
    #   return "hello world"
    
    def display2():
      return "hello world"
    
    display2()
    
    
    # After remove the dead code
   --------------------------------
    
    def display2():
      return "hello world"
  
    display2()
   ```
2. Import statements at the top of file

   Meaning that import statement should be at the top of file eadability, maintainability, and avoiding unexpected behaviors
   ```
   def display(radius):
     from math import pi
     return pi * radius * 2

   the CCD code for this:
   ----------------------
   from math import pi
   def display(radius):
     return pi * radius * 2
   ```
3. Error handling
   Without proper error handling, a program can crash or produce incorrect results,
   ```
   Example:
   --------
   
   print(result = 1/0) - It will give us ZeroDivisionError, and other code below of this would not be worked.

   To solve this - use Error handling
   ----------------------------------
   try:
     result = 1/0
   except Exception:
     print(e)
   finally:
     print("hello world")
     
   ```
4. Should be written Good Comments
   Comments are used to explain and describe the code.
   ```
   Example:
   -------
   # This function is used for calculating the sum of two number
   def addition(a, b):
     return a + b
   ```
5. DRY - Do not repeat yourself - write functions
  ```
  here, In the below code pi is used twice times. 
  radius = 5
  area = 3.14 * radius **2
  print(area)
  circle = 2 * 3.14 * radius
  print(circle)

  DRY: solve:
  ----------
  def area(radius)
    return 3.14 * radius **2

  def circle(radius)
    return 3.14 * radius **2
  ```
6. Explanatory variables
  Using well-named variables makes the code self-explanatory 
  ```
  Examples:
  ---------
  a = "Atikul Islam Sajib"
  b = 929199

  Using Explanatory variables:
  ---------------------------
  name = "Atikul Islam Sajib"
  matriculation = 929199
  ```
7. Tests - one assert statements per test
   It easier to pinpoint the source of failures and understand the scope of each test.
   ```
   Example:
   def display():
     assert ""Hello"+ " "+"world" == "Hello world"
   ```
8. Descriptive function names
   Descriptive function name gives the clearity of that function for users and it makes appropriate.
  ```
  Example:
  -------
  def a(d,e):
    return d + e

   After Descriptive function names:
  ----------------------------------
  def addition(num1, num2):
    return num1 + num2
  ```
9. Add docstrings to the functions and classes
  Add docstrings to the functions and classes makes the clearity and makes the code self-explanatory
  ```
  Example():
  ----------
  # This function provides the addition of two numbers. 
  def addition(num1, num2):
    return num1 + num2

  To more precise using doc strings:
  ----------------------------------
  # Calculate the addition of two numbers
  def addition(num1, num2):
  """
  Parameters:
     num1(int/float): The first number
     num2(int/float): The second number
   
     Returns:
     int/float: The sum of num1 and num2
  """
  ```
10. In functions and classes: Use parameter types and return types
    This indicates the type of parameters a function expects to receive and type of value it will return. 
  ```
    # Calculate the addition of two numbers
    def addition(num1:int, num2:int)->int:
    """
    Parameters:
    num1(int/float): The first number
    num2(int/float): The second number
  
    Returns:
    int/float: The sum of num1 and num2
  
    if __name__ == "__main__":
      print(addition(num1 = 10, num2 = 20))
    """
  ```
