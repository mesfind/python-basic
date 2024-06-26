---
title: Basics of python
teaching: 20
exercises: 20
objectives:
- Assign values to variables.
- Correctly trace value changes in programs that use scalar assignment.
questions:
- How can I store data in programs?
- What basic data types can I work with in Python?
- How can I create a new variable in Python?
- How do I use a function?
- Can I change the value associated with a variable after I create it?
keypoints:
- Use variables to store values.
- Use `print` to display values.
- Variables persist between cells.
- Variables must be created before they are used.
- Variables can be used in calculations.
- Use an index to get a single character from a string.
- Use a slice to get a substring.
- Use the built-in function `len` to find the length of a string.
- Python is case-sensitive.
- Use meaningful variable names.
- Basic data types in Python include integers, strings, and floating-point numbers.
- Use `variable = value` to assign a value to a variable in order to record it in memory.
- Variables are created on demand whenever a value is assigned to them.
- Use `print(something)` to display the value of `something`.
- Use `# some kind of explanation` to add comments to programs.
- Built-in functions are always available to use.
---

## Variables

Any Python interpreter can be used as a calculator:

```python
3 + 5 * 4
```

```output
23
```

This is great but not very interesting.
To do anything useful with data, we need to assign its value to a *variable*.
In Python, we can [assign](../learners/reference.md#assign) a value to a
[variable](../learners/reference.md#variable), using the equals sign `=`.
For example, we can track the weight of a patient who weighs 60 kilograms by
assigning the value `60` to a variable `weight_kg`:

```python
weight_kg = 60.3
```

From now on, whenever we use `weight_kg`, Python will substitute the value we assigned to
it. In layperson's terms, **a variable is a name for a value**.

In Python, variable names:

- can include letters, digits, and underscores
- cannot start with a digit
- are [case sensitive](../learners/reference.md#case-sensitive).

This means that, for example:

- `weight0` is a valid variable name, whereas `0weight` is not
- `weight` and `Weight` are different variables



## Use variables to store values.

- **Variables** are names for values.

- Variable names
  
  - can **only** contain letters, digits, and underscore `_` (typically used to separate words in long variable names)
  - cannot start with a digit
  - are **case sensitive** (age, Age and AGE are three different variables)

- The name should also be meaningful so you or another programmer know what it is

- Variable names that start with underscores like `__alistairs_real_age` have a special meaning
  so we won't do that until we understand the convention.

- In Python the `=` symbol assigns the value on the right to the name on the left.

- The variable is created when a value is assigned to it.

- Here, Python assigns an age to a variable `age`
  and a name in quotes to a variable `first_name`.
  
  ```python
  age = 42
  first_name = 'Abebe'
  ```

## Use `print` to display values.

- Python has a built-in function called `print` that prints things as text.
- Call the function (i.e., tell Python to run it) by using its name.
- Provide values to the function (i.e., the things to print) in parentheses.
- To add a string to the printout, wrap the string in single or double quotes.
- The values passed to the function are called **arguments**

```python
print(first_name, 'is', age, 'years old')
```

```output
Abebe is 42 years old
```

- `print` automatically puts a single space between items to separate them.
- And wraps around to a new line at the end.

## Variables must be created before they are used.

- If a variable doesn't exist yet, or if the name has been mis-spelled,
  Python reports an error. (Unlike some languages, which "guess" a default value.)

```python
print(last_name)
```

```error
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-1-c1fbb4e96102> in <module>()
----> 1 print(last_name)

NameError: name 'last_name' is not defined
```

- The last line of an error message is usually the most informative.
- We will look at error messages in detail [later](17-scope.md#reading-error-messages).



> ## Variables Persist Between Cells
> 
> Be aware that it is the *order* of execution of cells that is important in a Jupyter notebook, not the order
> in which they appear. Python will remember *all* the code that was run previously, including any variables you have
> defined, irrespective of the order in the notebook. Therefore if you define variables lower down the notebook and then
> (re)run cells further up, those defined further down will still be present. As an example, create two cells with the
> following content, in this order:
> 
> ```python
> print(myval)
> ```
> 
> ```python
> myval = 1
> ```
> 
> If you execute this in order, the first cell will give an error. However, if you run the first cell *after* the second
> cell it will print out `1`. To prevent confusion, it can be helpful to use the `Kernel` -> `Restart & Run All` option which
> clears the interpreter and runs everything from a clean slate going top to bottom.
{: .callout}


## Variables can be used in calculations.

- We can use variables in calculations just as if they were values.
  - Remember, we assigned the value `42` to `age` a few lines ago.

```python
age = age + 3
print('Age in three years:', age)
```

```output
Age in three years: 45
```

## Use an index to get a single character from a string.

- The characters (individual letters, numbers, and so on) in a string are
  ordered. For example, the string `'AB'` is not the same as `'BA'`. Because of
  this ordering, we can treat the string as a list of characters.
- Each position in the string (first, second, etc.) is given a number. This
  number is called an **index** or sometimes a subscript.
- Indices are numbered from 0.
- Use the position's index in square brackets to get the character at that
  position.

![A line of Python code, print(atom\_name[0]), demonstrates that using the zero index will output just the initial letter, in this case 'h' for helium.](../fig/2_indexing.svg)

```python
atom_name = 'helium'
print(atom_name[0])
```

```output
h
```

## Use a slice to get a substring.

- A part of a string is called a **substring**. A substring can be as short as a
  single character.
- An item in a list is called an element. Whenever we treat a string as if it
  were a list, the string's elements are its individual characters.
- A slice is a part of a string (or, more generally, a part of any list-like thing).
- We take a slice with the notation `[start:stop]`, where `start` is the integer
  index of the first element we want and `stop` is the integer index of
  the element *just after* the last element we want.
- The difference between `stop` and `start` is the slice's length.
- Taking a slice does not change the contents of the original string. Instead,
  taking a slice returns a copy of part of the original string.

```python
atom_name = 'sodium'
print(atom_name[0:3])
```

```output
sod
```

## Use the built-in function `len` to find the length of a string.

```python
print(len('helium'))
```

```output
6
```

- Nested functions are evaluated from the inside out,
  like in mathematics.

## Python is case-sensitive.

- Python thinks that upper- and lower-case letters are different,
  so `Name` and `name` are different variables.
- There are conventions for using upper-case letters at the start of variable names so we will use lower-case letters for now.

## Use meaningful variable names.

- Python doesn't care what you call variables as long as they obey the rules
  (alphanumeric characters and the underscore).

```python
flabadab = 42
ewr_422_yY = 'Mamo'
print(ewr_422_yY, 'is', flabadab, 'years old')
```

- Use meaningful variable names to help other people understand what the program does.
- The most important "other person" is your future self.


> ## Swapping Values
>
> Fill the table showing the values of the variables in this program *after* each statement is executed.
> 
> ~~~
> # Command  # Value of x   # Value of y   # Value of swap #
> x = 1.0    #              #              #               #
> y = 3.0    #              #              #               #
> swap = x   #              #              #               #
> x = y      #              #              #               #
> y = swap   #              #              #               #
> ~~~
> {: .python}
> 
> > ## Solution
> > 
> > ~~~
> > # Command  # Value of x   # Value of y   # Value of swap #
> > x = 1.0    # 1.0          # not defined  # not defined   #
> > y = 3.0    # 1.0          # 3.0          # not defined   #
> > swap = x   # 1.0          # 3.0          # 1.0           #
> > x = y      # 3.0          # 3.0          # 1.0           #
> > y = swap   # 3.0          # 1.0          # 1.0           #
> > ~~~
> > {: .output}
> {: .solution}
{: .challenge}

These three lines exchange the values in `x` and `y` using the `swap`
variable for temporary storage. This is a fairly common programming idiom.



> ## Predicting Values
> 
> What is the final value of `position` in the program below?
> (Try to predict the value without running the program,
> then check your prediction.)
> 
> ```python
> initial = 'left'
> position = initial
> initial = 'right'
> ```
>
> > ## Solution
> >
> > ```python
> > print(position)
> > ```
> >
> > ```output
> > left
> > ```
> {: .solution}
{: .challenge}

The `initial` variable is assigned the value `'left'`.
In the second line, the `position` variable also receives
the string value `'left'`. In third line, the `initial` variable is given the
value `'right'`, but the `position` variable retains its string value
of `'left'`.


> ## Challenge
> 
> If you assign `a = 123`,
> what happens if you try to get the second digit of `a` via `a[1]`?
> 
> > ## Solution
> > Numbers are not strings or sequences and Python will raise an error if you try to perform an index operation on a number. we will learn more about types and how to convert between different types. If you want the Nth digit of a number you can convert it into a string using the `str` built-in function and then perform an index operation on that string.
> > 
> > ```python
> > a = 123
> > print(a[1])
> > ```
> >
> > ```error
> > TypeError: 'int' object is not subscriptable
> > ```
> > 
> > ```python
> > a = str(123)
> > print(a[1])
> > ```
> >
> > ```output
> > 2
> > ```
> {: .solution}
{: .challenge}


> ## Choosing a Name
> 
> Which is a better variable name, `m`, `min`, or `minutes`?
> Why?
> Hint: think about which code you would rather inherit
> from someone who is leaving the lab:
> 
> 1. `ts = m * 60 + s`
> 2. `tot_sec = min * 60 + sec`
> 3. `total_seconds = minutes * 60 + seconds`
> 
> >  ## Solution
> >
> >`minutes` is better because `min` might mean something like "minimum"
> >(and actually is an existing built-in function in Python that we will cover later).
> {: .solution}
{: .challenge}




> ## Slicing practice
>
> What does the following program print?
> 
> ```python
> atom_name = 'carbon'
> print('atom_name[1:3] is:', atom_name[1:3])
> ```
> 
>
> > ## Solution
> >
> >```output
> >atom_name[1:3] is: ar
> >```
> {: .solution}
{: .challenge}


> ## Slicing concepts
> 
> Given the following string:
> 
> ```python
> species_name = "Acacia buxifolia"
> ```
> 
> What would these expressions return?
> 
> 1. `species_name[2:8]`
> 2. `species_name[11:]` (without a value after the colon)
> 3. `species_name[:4]` (without a value before the colon)
> 4. `species_name[:]` (just a colon)
> 5. `species_name[11:-3]`
> 6. `species_name[-5:-3]`
> 7. What happens when you choose a `stop` value which is out of range? (i.e., try `species_name[0:20]` or `species_name[:103]`)
> 
> > ## Solutions
> > 
> > 1. `species_name[2:8]` returns the substring `'acia b'`
> > 2. `species_name[11:]` returns the substring `'folia'`, from position 11 until the end
> > 3. `species_name[:4]` returns the substring `'Acac'`, from the start up to but not including position 4
> > 4. `species_name[:]` returns the entire string `'Acacia buxifolia'`
> > 5. `species_name[11:-3]` returns the substring `'fo'`, from the 11th position to the third last position
> > 6. `species_name[-5:-3]` also returns the substring `'fo'`, from the fifth last position to the third last
> > 7. If a part of the slice is out of range, the operation does not fail. `species_name[0:20]` gives the same result as `species_name[0:]`, and `species_name[:103]` gives the same result as `species_name[:]`
> {: .solution}  
{: .challenge}

## Types of data


Python knows various types of data. Three common ones are:

- integer numbers
- floating point numbers, and
- strings.

In the example above, variable `weight_kg` has an integer value of `60`.
If we want to more precisely track the weight of our patient,
we can use a floating point value by executing:

```python
weight_kg = 60.3
```

To create a string, we add single or double quotes around some text.
To identify and track a patient throughout our study,
we can assign each person a unique identifier by storing it in a string:

```python
patient_id = '001'
```

## Using Variables in Python

Once we have data stored with variable names, we can make use of it in calculations.
We may want to store our patient's weight in pounds as well as kilograms:

```python
weight_lb = 2.2 * weight_kg
```

We might decide to add a prefix to our patient identifier:

```python
patient_id = 'inflam_' + patient_id
```


## Built-in Python functions

To carry out common tasks with data and variables in Python,
the language provides us with several built-in [functions](../learners/reference.md#function).
To display information to the screen, we use the `print` function:

```python
print(weight_lb)
print(patient_id)
```

```output
132.66
inflam_001
```

When we want to make use of a function, referred to as calling the function,
we follow its name by parentheses. The parentheses are important:
if you leave them off, the function doesn't actually run!
Sometimes you will include values or variables inside the parentheses for the function to use.
In the case of `print`,
we use the parentheses to tell the function what value we want to display.
We will learn more about how functions work and how to create our own in later episodes.

We can display multiple things at once using only one `print` call:

```python
print(patient_id, 'weight in kilograms:', weight_kg)
```

```output
inflam_001 weight in kilograms: 60.3
```

We can also call a function inside of another
[function call](../learners/reference.md#function-call).
For example, Python has a built-in function called `type` that tells you a value's data type:

```python
print(type(60.3))
print(type(patient_id))
```

```output
<class 'float'>
<class 'str'>
```

Moreover, we can do arithmetic with variables right inside the `print` function:

```python
print('weight in pounds:', 2.2 * weight_kg)
```

```output
weight in pounds: 132.66
```

The above command, however, did not change the value of `weight_kg`:

```python
print(weight_kg)
```

```output
60.3
```

To change the value of the `weight_kg` variable, we have to
**assign** `weight_kg` a new value using the equals `=` sign:

```python
weight_kg = 65.0
print('weight in kilograms is now:', weight_kg)
```

```output
weight in kilograms is now: 65.0
```

:::::::::::::::::::::::::::::::::::::::::  callout

> ## Variables as Sticky Notes
> 
> A variable in Python is analogous to a sticky note with a name written on it: assigning a value to a variable is like putting that sticky note on a particular value.
> 
> ![](../fig/python-sticky-note-variables-01.svg){alt='Value of 65.0 with weight\_kg label stuck on it'}
> 
> Using this analogy, we can investigate how assigning a value to one variable does **not** change values of other, seemingly related, variables.  For example, let's store the subject's weight in pounds in its own variable:
> 
> ```python
> # There are 2.2 pounds per kilogram
> weight_lb = 2.2 * weight_kg
> print('weight in kilograms:', weight_kg, 'and in pounds:', weight_lb)
> ```
> 
> ```output
> weight in kilograms: 65.0 and in pounds: 143.0
> ```
{: .callout}

Everything in a line of code following the '#' symbol is a
[comment](../learners/reference.md#comment) that is ignored by Python.
Comments allow programmers to leave explanatory notes for other
programmers or their future selves.

![](../fig/python-sticky-note-variables-02.svg){alt='Value of 65.0 with weight\_kg label stuck on it, and value of 143.0 with weight\_lb label stuck on it'}

Similar to above, the expression `2.2 * weight_kg` is evaluated to `143.0`,
and then this value is assigned to the variable `weight_lb` (i.e. the sticky
note `weight_lb` is placed on `143.0`). At this point, each variable is
"stuck" to completely distinct and unrelated values.

Let's now change `weight_kg`:

```python
weight_kg = 100.0
print('weight in kilograms is now:', weight_kg, 'and weight in pounds is still:', weight_lb)
```

```output
weight in kilograms is now: 100.0 and weight in pounds is still: 143.0
```

![](../fig/python-sticky-note-variables-03.svg){alt='Value of 100.0 with label weight\_kg stuck on it, and value of 143.0 with label weight\_lbstuck on it'}

Since `weight_lb` doesn't "remember" where its value comes from,
it is not updated when we change `weight_kg`.



> ## Check Your Understanding
> 
> What values do the variables `mass` and `age` have after each of the following statements?
> Test your answer by executing the lines.
> 
> ```python
> mass = 47.5
> age = 122
> mass = mass * 2.0
> age = age - 20
> ```
> 
> > ## Solution
> > 
> > ```output
> > `mass` holds a value of 47.5, `age` does not exist
> > `mass` still holds a value of 47.5, `age` holds a value of 122
> > `mass` now has a value of 95.0, `age`'s value is still 122
> > `mass` still has a value of 95.0, `age` now holds 102
> > ```
> {: .solution}
>
> ## Sorting Out References
> 
> Python allows you to assign multiple values to multiple variables in one line by separating
> the variables and values with commas. What does the following program print out?
> 
> ```python
> first, second = 'Grace', 'Hopper'
> third, fourth = second, first
> print(third, fourth)
> ```
>
> > ## Solution
> > 
> > ```output
> > Hopper Grace
> > ```
> {: .solution}
>
> ## Seeing Data Types
> 
> What are the data types of the following variables?
> 
> ```python
> planet = 'Earth'
> apples = 5
> distance = 10.5
> ```
> 
> > ## Solution
> > 
> > ```python
> > print(type(planet))
> > print(type(apples))
> > print(type(distance))
> > ```
> > 
> > ```output
> > <class 'str'>
> > <class 'int'>
> > <class 'float'>
> > ```
> {: .solution} 
{: .challenge}
