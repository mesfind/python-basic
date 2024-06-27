---
title: Data Types and Built-in Functions
teaching: 25
exercises: 20
objectives:
- Explain key differences between integers and floating point numbers.
- Explain key differences between numbers and character strings.
- Use built-in functions to convert between integers, floating point numbers, and strings.
- Explain the purpose of functions.
- Correctly call built-in Python functions.
- Correctly nest calls to built-in functions.
- Use help to display documentation for built-in functions.
- Correctly describe situations in which SyntaxError and NameError occur.
questions:
- What kinds of data do programs store?
- How can I convert one type to another?
- How can I use built-in functions?
- How can I find out what they do?
- What kind of errors can occur in programs?
keypoints:
- Every value has a type.
- Use the built-in function `type` to find the type of a value.
- Types control what operations can be done on values.
- Strings can be added and multiplied.
- Strings have a length (but numbers don't).
- Must convert numbers to strings or vice versa when operating on them.
- Can mix integers and floats freely in operations.
- Variables only change value when something is assigned to them.
- Use comments to add documentation to programs.
- A function may take zero or more arguments.
- Commonly-used built-in functions include `max`, `min`, and `round`.
- Functions may only work for certain (combinations of) arguments.
- Functions may have default values for some arguments.
- Use the built-in function `help` to get help for a function.
- The Jupyter Notebook has two ways to get help.
- Every function returns something.
- Python reports a syntax error when it can't understand the source of a program.
- Python reports a runtime error when something goes wrong while a program is executing.
- Fix syntax errors by reading the source code, and runtime errors by tracing the program's execution.
---


## Every value has a type.

- Every value in a program has a specific type.
- Integer (`int`): represents positive or negative whole numbers like 3 or -512.
- Floating point number (`float`): represents real numbers like 3.14159 or -2.5.
- Character string (usually called "string", `str`): text.
  - Written in either single quotes or double quotes (as long as they match).
  - The quote marks aren't printed when the string is displayed.

## Use the built-in function `type` to find the type of a value.

- Use the built-in function `type` to find out what type a value has.
- Works on variables as well.
  - But remember: the *value* has the type --- the *variable* is just a label.

```python
print(type(52))
```

```output
<class 'int'>
```

```python
fitness = 'average'
print(type(fitness))
```

```output
<class 'str'>
```

## Types control what operations (or methods) can be performed on a given value.

- A value's type determines what the program can do to it.

```python
print(5 - 3)
```

```output
2
```

```python
print('hello' - 'h')
```

```error
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-2-67f5626a1e07> in <module>()
----> 1 print('hello' - 'h')

TypeError: unsupported operand type(s) for -: 'str' and 'str'
```

## You can use the "+" and "\*" operators on strings.

- "Adding" character strings concatenates them.

```python
full_name = 'Abebe' + ' ' + 'Geresu'
print(full_name)
```

```output
Abebe Geresu
```

- Multiplying a character string by an integer *N* creates a new string that consists of that character string repeated  *N* times.
  - Since multiplication is repeated addition.

```python
separator = '=' * 10
print(separator)
```

```output
==========
```

## Strings have a length (but numbers don't).

- The built-in function `len` counts the number of characters in a string.

```python
print(len(full_name))
```

```output
11
```

- But numbers don't have a length (not even zero).

```python
print(len(52))
```

```error
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-3-f769e8e8097d> in <module>()
----> 1 print(len(52))

TypeError: object of type 'int' has no len()
```

## Must convert numbers to strings or vice versa when operating on them. {#convert-numbers-and-strings}

- Cannot add numbers and strings.

```python
print(1 + '2')
```

```error
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-4-fe4f54a023c6> in <module>()
----> 1 print(1 + '2')

TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

- Not allowed because it's ambiguous: should `1 + '2'` be `3` or `'12'`?
- Some types can be converted to other types by using the type name as a function.

```python
print(1 + int('2'))
print(str(1) + '2')
```

```output
3
12
```

## Can mix integers and floats freely in operations.

- Integers and floating-point numbers can be mixed in arithmetic.
  - Python 3 automatically converts integers to floats as needed.

```python
print('half is', 1 / 2.0)
print('three squared is', 3.0 ** 2)
```

```output
half is 0.5
three squared is 9.0
```

## Variables only change value when something is assigned to them.

- If we make one cell in a spreadsheet depend on another,
  and update the latter,
  the former updates automatically.
- This does **not** happen in programming languages.

```python
variable_one = 1
variable_two = 5 * variable_one
variable_one = 2
print('first is', variable_one, 'and second is', variable_two)
```

```output
first is 2 and second is 5
```

- The computer reads the value of `variable_one` when doing the multiplication,
  creates a new value, and assigns it to `variable_two`.
- Afterwards, the value of `variable_two` is set to the new value and *not dependent on `variable_one`* so its value
  does not automatically change when `variable_one` changes.



> ## Fractions
> 
> What type of value is 3.4?
> How can you find out?
> 
> > ## Solution
> > 
> > It is a floating-point number (often abbreviated "float").
> > It is possible to find out by using the built-in function `type()`.
> > 
> > ```python
> > print(type(3.4))
> >```
> >
> > ```output
> ><class 'float'>
> >```
> {: .solution}
{: .challenge}


> ## Automatic Type Conversion
> 
> What type of value is 3.25 + 4?
> 
> > ## Solution
> > 
> > It is a float:
> > integers are automatically converted to floats as necessary.
> > 
> > ```python
> > result = 3.25 + 4
> > print(result, 'is', type(result))
> > ```
> > 
> > ```output
> > 7.25 is <class 'float'>
> > ```
> {: .solution}
{: .challenge}


> ## Choose a Type
> 
> - What type of value (integer, floating point number, or character string)
> - would you use to represent each of the following?  Try to come up with more than one good answer for each problem.  For example, in  # 1, when would counting days with a floating point variable make more sense than using an integer?
> 
> 1. Number of days since the start of the year.
> 2. Time elapsed from the start of the year until now in days.
> 3. Serial number of a piece of lab equipment.
> 4. A lab specimen's age
> 5. Current population of a city.
> 6. Average population of a city over time.
> 
> > ## Solution
> > 
> > The answers to the questions are:
> > 
> > 1. Integer, since the number of days would lie between 1 and 365.
> > 2. Floating point, since fractional days are required
> > 3. Character string if serial number contains letters and numbers, otherwise integer if the serial number consists only of numerals
> > 4. This will vary! How do you define a specimen's age? whole days since collection (integer)? date and time (string)?
> > 5. Choose floating point to represent population as large aggregates (eg millions), or integer to represent population in units of individuals.
> > 6. Floating point number, since an average is likely to have a fractional part.
> {: .solution} 
{: .challenge}  



> ## Division Types
> 
> In Python 3, the `//` operator performs integer (whole-number) floor division, the `/` operator performs floating-point
> division, and the `%` (or *modulo*) operator calculates and returns the remainder from integer division:
> 
> ```python
> print('5 // 3:', 5 // 3)
> print('5 / 3:', 5 / 3)
> print('5 % 3:', 5 % 3)
> ```
> 
> ```output
> 5 // 3: 1
> 5 / 3: 1.6666666666666667
> 5 % 3: 2
> ```
> 
> If `num_subjects` is the number of subjects taking part in a study, and `num_per_survey` is the number that can take part in a single survey,
> write an expression that calculates the number of surveys needed to reach everyone once.
>
> > ## Solution
> > 
> > We want the minimum number of surveys that reaches everyone once, which is the rounded up value of `num_subjects/ num_per_survey`. This is equivalent to performing a floor division with `//` and adding 1. Before the division we need to subtract 1 from the number of subjects to deal with the case where `num_subjects` is evenly divisible by `num_per_survey`.
> > 
> > ```python
> > num_subjects = 600
> > num_per_survey = 42
> > num_surveys = (num_subjects - 1) // num_per_survey + 1
> > 
> > print(num_subjects, 'subjects,', num_per_survey, 'per survey:', num_surveys)
> > ```
> > 
> > ```output
> > 600 subjects, 42 per survey: 15
> > ```
> {: .solution}
{: .challenge}



> ## Strings to Numbers
> 
> Where reasonable, `float()` will convert a string to a floating point number, and `int()` will convert a floating point number to an integer:
> 
> ```python
> print("string to float:", float("3.4"))
> print("float to int:", int(3.4))
> ```
> 
> ```output
> string to float: 3.4
> float to int: 3
> ```
> 
> If the conversion doesn't make sense, however, an error message will occur.
> 
> ```python
> print("string to float:", float("Hello world!"))
> ```
> 
> ```error
> ---------------------------------------------------------------------------
> ValueError                                Traceback (most recent call last)
> <ipython-input-5-df3b790bf0a2> in <module>
> ----> 1 print("string to float:", float("Hello world!"))
> 
> ValueError: could not convert string to float: 'Hello world!'
> ```
> 
> Given this information, what do you expect the following program to do?
> 
> What does it actually do?
> 
> Why do you think it does that?
> ```python
> print("fractional string to int:", int("3.4"))
> ```
>
> > ## Solution
> > 
> > What do you expect this program to do? It would not be so unreasonable to expect the Python 3 `int` command to
> > convert the string "3.4" to 3.4 and an additional type conversion to 3. After all, Python 3 performs a lot of other magic - isn't that part of its charm?
> > 
> > ```python
> > int("3.4")
> > ```
> > 
> > ```output
> > ---------------------------------------------------------------------------
> > ValueError                                Traceback (most recent call last)
> > <ipython-input-2-ec6729dfccdc> in <module>
> > ----> 1 int("3.4")
> > ValueError: invalid literal for int() with base 10: '3.4'
> > ```
> > 
> > However, Python 3 throws an error. Why? To be consistent, possibly. If you ask Python to perform two consecutive
> > typecasts, you must convert it explicitly in code.
> > 
> > ```python
> > int(float("3.4"))
> > ```
> > 
> > ```output
> > 3
> > ```
> {: .solution}
{: .challenge}




## Use comments to add documentation to programs.

```python
# This sentence isn't executed by Python.
adjustment = 0.5   # Neither is this - anything after '#' is ignored.
```

## A function may take zero or more arguments.

- We have seen some functions already --- now let's take a closer look.
- An *argument* is a value passed into a function.
- `len` takes exactly one.
- `int`, `str`, and `float` create a new value from an existing one.
- `print` takes zero or more.
- `print` with no arguments prints a blank line.
  - Must always use parentheses, even if they're empty,
    so that Python knows a function is being called.

```python
print('before')
print()
print('after')
```

```output
before

after
```

## Every function returns something.

- Every function call produces some result.
- If the function doesn't have a useful result to return,
  it usually returns the special value `None`. `None` is a Python
  object that stands in anytime there is no value.

```python
result = print('example')
print('result of print is', result)
```

```output
example
result of print is None
```

## Commonly-used built-in functions include `max`, `min`, and `round`.

- Use `max` to find the largest value of one or more values.
- Use `min` to find the smallest.
- Both work on character strings as well as numbers.
  - "Larger" and "smaller" use (0-9, A-Z, a-z) to compare letters.

```python
print(max(1, 2, 3))
print(min('a', 'A', '0'))
```

```output
3
0
```

## Functions may only work for certain (combinations of) arguments.

- `max` and `min` must be given at least one argument.
  - "Largest of the empty set" is a meaningless question.
- And they must be given things that can meaningfully be compared.

```python
print(max(1, 'a'))
```

```error
TypeError                                 Traceback (most recent call last)
<ipython-input-52-3f049acf3762> in <module>
----> 1 print(max(1, 'a'))

TypeError: '>' not supported between instances of 'str' and 'int'
```

## Functions may have default values for some arguments.

- `round` will round off a floating-point number.
- By default, rounds to zero decimal places.

```python
round(3.712)
```

```output
4
```

- We can specify the number of decimal places we want.

```python
round(3.712, 1)
```

```output
3.7
```

## Functions attached to objects are called methods

- Functions take another form that will be common in the pandas episodes.
- Methods have parentheses like functions, but come after the variable.
- Some methods are used for internal Python operations, and are marked with double underlines.

```python
my_string = 'Hello world!'  # creation of a string object 

print(len(my_string))       # the len function takes a string as an argument and returns the length of the string

print(my_string.swapcase()) # calling the swapcase method on the my_string object

print(my_string.__len__())  # calling the internal __len__ method on the my_string object, used by len(my_string)

```

```output
12
hELLO WORLD!
12
```

- You might even see them chained together.  They operate left to right.

```python
print(my_string.isupper())          # Not all the letters are uppercase
print(my_string.upper())            # This capitalizes all the letters

print(my_string.upper().isupper())  # Now all the letters are uppercase
```

```output
False
HELLO WORLD
True
```

## Use the built-in function `help` to get help for a function.

- Every built-in function has online documentation.

```python
help(round)
```

```output
Help on built-in function round in module builtins:

round(number, ndigits=None)
    Round a number to a given precision in decimal digits.
    
    The return value is an integer if ndigits is omitted or None.  Otherwise
    the return value has the same type as the number.  ndigits may be negative.
```

## The Jupyter Notebook has two ways to get help.

- Option 1: Place the cursor near where the function is invoked in a cell
  (i.e., the function name or its parameters),
  - Hold down <kbd>Shift</kbd>, and press <kbd>Tab</kbd>.
  - Do this several times to expand the information returned.
- Option 2: Type the function name in a cell with a question mark after it. Then run the cell.

## Python reports a syntax error when it can't understand the source of a program.

- Won't even try to run the program if it can't be parsed.

```python
# Forgot to close the quote marks around the string.
name = 'Feng
```

```error
  File "<ipython-input-56-f42768451d55>", line 2
    name = 'Feng
                ^
SyntaxError: EOL while scanning string literal
```

```python
# An extra '=' in the assignment.
age = = 52
```

```error
  File "<ipython-input-57-ccc3df3cf902>", line 2
    age = = 52
          ^
SyntaxError: invalid syntax
```

- Look more closely at the error message:

```python
print("hello world"
```

```error
  File "<ipython-input-6-d1cc229bf815>", line 1
    print ("hello world"
                        ^
SyntaxError: unexpected EOF while parsing
```

- The message indicates a problem on first line of the input ("line 1").
  - In this case the "ipython-input" section of the file name tells us that
    we are working with input into IPython,
    the Python interpreter used by the Jupyter Notebook.
- The `-6-` part of the filename indicates that
  the error occurred in cell 6 of our Notebook.
- Next is the problematic line of code,
  indicating the problem with a `^` pointer.

## Python reports a runtime error when something goes wrong while a program is executing. {#runtime-error}

```python
age = 53
remaining = 100 - aege # mis-spelled 'age'
```

```error
NameError                                 Traceback (most recent call last)
<ipython-input-59-1214fb6c55fc> in <module>
      1 age = 53
----> 2 remaining = 100 - aege # mis-spelled 'age'

NameError: name 'aege' is not defined
```

- Fix syntax errors by reading the source and runtime errors by tracing execution.


> ## What Happens When
>
> 1. Explain in simple terms the order of operations in the following program:
>  when does the addition happen, when does the subtraction happen,  when is each function called, etc.
> 2. What is the final value of `radiance`?
>
> ```python
> radiance = 1.0
> radiance = max(2.1, 2.0 + min(radiance, 1.1 * radiance - 0.5))
> ```
>
> >## Solution
> >
> > 1. Order of operations:
> >   1. `1.1 * radiance = 1.1`
> >   2. `1.1 - 0.5 = 0.6`
> >   3. `min(radiance, 0.6) = 0.6`
> >   4. `2.0 + 0.6 = 2.6`
> >   5. `max(2.1, 2.6) = 2.6`
> > 2. At the end, `radiance = 2.6`
> {: .solution}
{: .challenge}


