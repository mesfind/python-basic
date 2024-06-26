---
title: Analyzing Data with Numpy
teaching: 40
exercises: 20
objectives:
- Use a library function to get a list of filenames that match a wildcard pattern.
- Write a `for` loop to process multiple files.
- Explain what a library is and what libraries are used for.
- Import a Python library and use the functions it contains.
- Read tabular data from a file into a program.
- Select individual values and subsections from data.
- Perform operations on arrays of data.
questions:
- How can I do the same operations on many different files?
- How can I process tabular data files in Python?
keypoints: 
- Import a library into a program using `import libraryname`.
- Use the `numpy` library to work with arrays in Python.
- The expression `array.shape` gives the shape of an array.
- Use `array[x, y]` to select a single element from a 2D array.
- Array indices start at 0, not 1.
- Use `low:high` to specify a `slice` that includes the indices from `low` to `high-1`.
- Use `# some kind of explanation` to add comments to programs.
- Use `np.mean(array)`, `np.amax(array)`, and `np.amin(array)` to calculate simple statistics.
- Use `np.mean(array, axis=0)` or `np.mean(array, axis=1)` to calculate statistics across the specified axis.
- Use `glob(pattern)` to create a list of files whose names match a pattern.
- Use `*` in a pattern to match zero or more characters, and `?` to match any single character.
---


Words are useful, but what's more useful are the sentences and stories we build with them. Similarly, while a lot of powerful, general tools are built into Python,specialized tools built up from these basic units live in [libraries](../learners/reference.md#library) that can be called upon when needed.

## Loading data into Python

To begin processing the clinical trial inflammation data, we need to load it into Python.
We can do that using a library called
[NumPy](https://numpy.org/doc/stable "NumPy Documentation"), which stands for Numerical Python.
In general, you should use this library when you want to do fancy things with lots of numbers,
especially if you have matrices or arrays. To tell Python that we'd like to start using NumPy,
we need to [import](../learners/reference.md#import) it:

~~~
import numpy as np
~~~
{: .python}

Importing a library is like getting a piece of lab equipment out of a storage locker and setting it
up on the bench. Libraries provide additional functionality to the basic Python package, much like
a new piece of equipment adds functionality to a lab space. Just like in the lab, importing too
many libraries can sometimes complicate and slow down your programs - so we only import what we
need for each program.

Once we've imported the library, we can ask the library to read our data file for us:

~~~
np.loadtxt(fname='data/inflammation-01.csv', delimiter=',')
~~~
{: .python}

~~~
array([[ 0.,  0.,  1., ...,  3.,  0.,  0.],
       [ 0.,  1.,  2., ...,  1.,  0.,  1.],
       [ 0.,  1.,  1., ...,  2.,  1.,  1.],
       ...,
       [ 0.,  1.,  1., ...,  1.,  1.,  1.],
       [ 0.,  0.,  0., ...,  0.,  2.,  0.],
       [ 0.,  0.,  1., ...,  1.,  1.,  0.]])
~~~
{: .output}

The expression `np.loadtxt(...)` is a
[function call](../learners/reference.md#function-call)
that asks Python to run the [function](../learners/reference.md#function) `loadtxt` which
belongs to the `numpy` library.
The dot notation in Python is used most of all as an object attribute/property specifier or for invoking its method. `object.property` will give you the object.property value,
`object_name.method()` will invoke on object\_name method.

As an example, Abebe Bekele is the John that belongs to the Smith family. We could use the dot notation to write his name `abebe.bekele`, just as `loadtxt` is a function that belongs to the `numpy` library.

`numpy.loadtxt` has two [parameters](../learners/reference.md#parameter): the name of the file
we want to read and the [delimiter](../learners/reference.md#delimiter) that separates values
on a line. These both need to be character strings
(or [strings](../learners/reference.md#string) for short), so we put them in quotes.

Since we haven't told it to do anything else with the function's output,
the [notebook](../learners/reference.md#notebook) displays it.
In this case, that output is the data we just loaded. By default, only a few rows and columns are shown (with `...` to omit elements when displaying big arrays). Note that, to save space when displaying NumPy arrays, Python does not show us trailing zeros, so `1.0` becomes `1.`.

Our call to `np.loadtxt` read our file but didn't save the data in memory. To do that, we need to assign the array to a variable. In a similar manner to how we assign a single value to a variable, we can also assign an array of values to a variable using the same syntax. Let's re-run `np.loadtxt` and save the returned data:

~~~
data = np.loadtxt(fname='data/inflammation-01.csv', delimiter=',')
~~~
{: .python}

This statement doesn't produce any output because we've assigned the output to the variable `data`.
If we want to check that the data have been loaded,
we can print the variable's value:

~~~
print(data)
~~~
{: .python}

~~~
[[ 0.  0.  1. ...,  3.  0.  0.]
 [ 0.  1.  2. ...,  1.  0.  1.]
 [ 0.  1.  1. ...,  2.  1.  1.]
 ...,
 [ 0.  1.  1. ...,  1.  1.  1.]
 [ 0.  0.  0. ...,  0.  2.  0.]
 [ 0.  0.  1. ...,  1.  1.  0.]]
~~~
{: .output}

Now that the data are in memory,
we can manipulate them.
First,
let's ask what [type](../learners/reference.md#type) of thing `data` refers to:

~~~
print(type(data))
~~~
{: .python}

~~~
<class 'numpy.ndarray'>
~~~
{: .output}

The output tells us that `data` currently refers to an N-dimensional array, the functionality for which is provided by the NumPy library. These data correspond to arthritis patients' inflammation. The rows are the individual patients, and the columns are their daily inflammation measurements.



> ## Data Type
> 
> A Numpy array contains one or more elements of the same type. The `type` function will only tell you that a variable is a NumPy array but won't tell you the type of
thing inside the array. We can find out the type of the data contained in the NumPy array.
> 
> ~~~
> print(data.dtype)
> ~~~
> {: .python}
> 
> ~~~
> float64
> ~~~
> {: .python}
> 
> This tells us that the NumPy array's elements are [floating-point numbers](../learners/reference.md#floating-point-number).
{: .callout}



With the following command, we can see the array's [shape](../learners/reference.md#shape):

~~~
print(data.shape)
~~~
{: .python}

~~~
(60, 40)
~~~
{: .output}

The output tells us that the `data` array variable contains 60 rows and 40 columns. When we
created the variable `data` to store our arthritis data, we did not only create the array; we also
created information about the array, called [members](../learners/reference.md#member) or
attributes. This extra information describes `data` in the same way an adjective describes a noun.
`data.shape` is an attribute of `data` which describes the dimensions of `data`. We use the same
dotted notation for the attributes of variables that we use for the functions in libraries because
they have the same part-and-whole relationship.

If we want to get a single number from the array, we must provide an
[index](../learners/reference.md#index) in square brackets after the variable name, just as we
do in math when referring to an element of a matrix.  Our inflammation data has two dimensions, so
we will need to use two indices to refer to one specific value:

~~~
print('first value in data:', data[0, 0])
~~~
{: .python}

~~~
first value in data: 0.0
~~~
{: .output}

~~~
print('middle value in data:', data[29, 19])
~~~
{: .python}

~~~
middle value in data: 16.0
~~~
{: .output}

The expression `data[29, 19]` accesses the element at row 30, column 20. While this expression may not surprise you,`data[0, 0]` might.Programming languages like Fortran, MATLAB and R start counting at 1 because that's what human beings have done for thousands of years.Languages in the C family (including C++, Java, Perl, and Python) count from 0 because it represents an offset from the first value in the array (the second value is offset by one index from the first value). This is closer to the way that computers represent arrays (if you are interested in the historical reasons behind counting indices from zero, you can read [Mike Hoye's blog post](https://exple.tive.org/blarg/2013/10/22/citation-needed/)). As a result,
if we have an M×N array in Python, its indices go from 0 to M-1 on the first axis and 0 to N-1 on the second. It takes a bit of getting used to, but one way to remember the rule is that the index is how many steps we have to take from the start to get the item we want.

!['data' is a 3 by 3 numpy array containing row 0: \['A', 'B', 'C'\], row 1: \['D', 'E', 'F'\], androw 2: \['G', 'H', 'I'\]. Starting in the upper left hand corner, data\[0, 0\] = 'A', data\[0, 1\] = 'B',data\[0, 2\] = 'C', data\[1, 0\] = 'D', data\[1, 1\] = 'E', data\[1, 2\] = 'F', data\[2, 0\] = 'G',data\[2, 1\] = 'H', and data\[2, 2\] = 'I', in the bottom right hand corner.](../fig/python-zero-index.svg)



> ## In the Corner
> 
> What may also surprise you is that when Python displays an array, it shows the element with index `[0, 0]` in the upper left corner rather than the lower left. This is consistent with the way mathematicians draw matrices but different from the Cartesian coordinates. The indices are (row, column) instead of (column, row) for the same reason, which can be confusing when plotting data.
{: .callout}


## Slicing data

An index like `[30, 20]` selects a single element of an array,
but we can select whole sections as well.
For example,
we can select the first ten days (columns) of values
for the first four patients (rows) like this:

~~~
print(data[0:4, 0:10])
~~~
{: .python}

~~~
[[ 0.  0.  1.  3.  1.  2.  4.  7.  8.  3.]
 [ 0.  1.  2.  1.  2.  1.  3.  2.  2.  6.]
 [ 0.  1.  1.  3.  3.  2.  6.  2.  5.  9.]
 [ 0.  0.  2.  0.  4.  2.  2.  1.  6.  7.]]
~~~
{: .output}

The [slice](../learners/reference.md#slice) `0:4` means, "Start at index 0 and go up to,
but not including, index 4". Again, the up-to-but-not-including takes a bit of getting used to,
but the rule is that the difference between the upper and lower bounds is the number of values in
the slice.

We don't have to start slices at 0:

~~~
print(data[5:10, 0:10])
~~~
{: .python}

~~~
[[ 0.  0.  1.  2.  2.  4.  2.  1.  6.  4.]
 [ 0.  0.  2.  2.  4.  2.  2.  5.  5.  8.]
 [ 0.  0.  1.  2.  3.  1.  2.  3.  5.  3.]
 [ 0.  0.  0.  3.  1.  5.  6.  5.  5.  8.]
 [ 0.  1.  1.  2.  1.  3.  5.  3.  5.  8.]]
~~~
{: .output}
We also don't have to include the upper and lower bound on the slice.  If we don't include the lower
bound, Python uses 0 by default; if we don't include the upper, the slice runs to the end of the
axis, and if we don't include either (i.e., if we use ':' on its own), the slice includes
everything:

~~~
small = data[:3, 36:]
print('small is:')
print(small)
~~~
{: .python}

The above example selects rows 0 through 2 and columns 36 through to the end of the array.

~~~
small is:
[[ 2.  3.  0.  0.]
 [ 1.  1.  0.  1.]
 [ 2.  2.  1.  1.]]
~~~
{: .output}

## Analyzing data

NumPy has several useful functions that take an array as input to perform operations on its values. If we want to find the average inflammation for all patients on all days, for example, we can ask NumPy to compute `data`'s mean value:

~~~
print(np.mean(data))
~~~
{: .python}

~~~
6.14875
~~~
{: .output}

`mean` is a [function](../learners/reference.md#function) that takes an array as an [argument](../learners/reference.md#argument).



> ## Not All Functions Have Input
> 
> Generally, a function uses inputs to produce outputs. However, some functions produce outputs without needing any input. For example, checking the current time doesn't require any input.
> 
> ~~~
> import time
> print(time.ctime())
> ~~~
> {: .python}
> 
> ~~~
> Sat Mar 26 13:07:33 2016
> ~~~
> {: .python}
> 
> For functions that don't take in any arguments, we still need parentheses (`()`) to tell Python to go and do something for us.
{: .callout}



Let's use three other NumPy functions to get some descriptive values about the dataset. We'll also use multiple assignment, a convenient Python feature that will enable us to do this all in one line.

~~~
maxval, minval, stdval = np.amax(data), np.amin(data), np.std(data)

print('maximum inflammation:', maxval)
print('minimum inflammation:', minval)
print('standard deviation:', stdval)
~~~
{: .python}

Here we've assigned the return value from `np.amax(data)` to the variable `maxval`, the value
from `np.amin(data)` to `minval`, and so on.

~~~
maximum inflammation: 20.0
minimum inflammation: 0.0
standard deviation: 4.61383319712
~~~
{: .output}


> ## Mystery Functions in IPython
> 
> How did we know what functions NumPy has and how to use them? If you are working in IPython or in a Jupyter Notebook, there is an easy way to find out. If you type the name of something followed by a dot, then you can use [tab completion](../learners/reference.md#tab-completion)(e.g. type `np.` and then press <kbd>Tab</kbd>) to see a list of all functions and attributes that you can use. After selecting one, you can also add a question mark (e.g. `np.cumprod?`), and IPython will return an explanation of the method! This is the same as doing `help(np.cumprod)`. Similarly, if you are using the "plain vanilla" Python interpreter, you can type `numpy.` and press the <kbd>Tab</kbd> key twice for a listing of what is available. You can then use the `help()` function to see an explanation of the function you're interested in,
for example: `help(np.cumprod)`.
{: .callout}



> ## Confusing Function Names
> 
> One might wonder why the functions are called `amax` and `amin` and not `max` and `min` or why the other is called `mean` and not `amean`. The package `numpy` does provide functions `max` and `min` that are fully equivalent to `amax` and `amin`, but they share a name with standard library functions `max` and `min` that come with Python itself. Referring to the functions like we did above, that is `numpy.max` for example, does not cause problems, but there are other ways to refer to them that could. In addition, text editors might highlight (color) these functions like standard library function, even though they belong to NumPy, which can be confusing and lead to errors. Since there is no function called `mean` in the standard library, there is no function called `amean`.
{: .callout}


When analyzing data, though,we often want to look at variations in statistical values,
such as the maximum inflammation per patientor the average inflammation per day.One way to do this is to create a new temporary array of the data we want,then ask it to do the calculation:

~~~
patient_0 = data[0, :] # 0 on the first axis (rows), everything on the second (columns)
print('maximum inflammation for patient 0:', np.amax(patient_0))
~~~
{: .python}

~~~
maximum inflammation for patient 0: 18.0
~~~
{: .output}

We don't actually need to store the row in a variable of its own.
Instead, we can combine the selection and the function call:

~~~
print('maximum inflammation for patient 2:', np.amax(data[2, :]))
~~~
{: .python}

~~~
maximum inflammation for patient 2: 19.0
~~~
{: .output}

What if we need the maximum inflammation for each patient over all days (as in the
next diagram on the left) or the average for each day (as in the
diagram on the right)? As the diagram below shows, we want to perform the
operation across an axis:

![Per-patient maximum inflammation is computed row-wise across all columns usingnumpy.amax(data, axis=1). Per-day average inflammation is computed column-wise across all rows usingnumpy.mean(data, axis=0).](../fig/python-operations-across-axes.png)

To support this functionality,
most array functions allow us to specify the axis we want to work on.
If we ask for the average across axis 0 (rows in our 2D example),
we get:

~~~
print(np.mean(data, axis=0))
~~~
{: .python}

~~~
[  0.           0.45         1.11666667   1.75         2.43333333   3.15
   3.8          3.88333333   5.23333333   5.51666667   5.95         5.9
   8.35         7.73333333   8.36666667   9.5          9.58333333
  10.63333333  11.56666667  12.35        13.25        11.96666667
  11.03333333  10.16666667  10.           8.66666667   9.15         7.25
   7.33333333   6.58333333   6.06666667   5.95         5.11666667   3.6
   3.3          3.56666667   2.48333333   1.5          1.13333333
   0.56666667]
~~~
{: .output}

As a quick check,
we can ask this array what its shape is:

~~~
print(np.mean(data, axis=0).shape)
~~~
{: .python}

~~~
(40,)
~~~
{: .output}

The expression `(40,)` tells us we have an N×1 vector,
so this is the average inflammation per day for all patients.
If we average across axis 1 (columns in our 2D example), we get:

~~~
print(np.mean(data, axis=1))
~~~
{: .python}

~~~
[ 5.45   5.425  6.1    5.9    5.55   6.225  5.975  6.65   6.625  6.525
  6.775  5.8    6.225  5.75   5.225  6.3    6.55   5.7    5.85   6.55
  5.775  5.825  6.175  6.1    5.8    6.425  6.05   6.025  6.175  6.55
  6.175  6.35   6.725  6.125  7.075  5.725  5.925  6.15   6.075  5.75
  5.975  5.725  6.3    5.9    6.75   5.925  7.225  6.15   5.95   6.275  5.7
  6.1    6.825  5.975  6.725  5.7    6.25   6.4    7.05   5.9  ]
~~~
{: .output}

which is the average inflammation per patient across all days.



> ## Slicing Strings
> 
> A section of an array is called a [slice](../learners/reference.md#slice). We can take slices of character strings as well:
> 
> ~~~
> element = 'oxygen'
> print('first three characters:', element[0:3])
> print('last three characters:', element[3:6])
> ~~~
> {: .python}
> 
> ~~~
> first three characters: oxy
> last three characters: gen
> ~~~
> {: .output}
> 
> - What is the value of `element[:4]`?
> - What about `element[4:]`?
> - Or `element[:]`?
>
> > ## Solution
> >
> > ~~~
> > oxyg
> > en
> > oxygen
> > ~~~
> > {: .output}
> {: .solution}
> 
> What is `element[-1]`?
> What is `element[-2]`?
> 
> 
> >## Solution
> >
> >```output
> >n
> >e
> >```
> {: .solution}
> 
> Given those answers,
> explain what `element[1:-1]` does.
> 
> > ## Solution
> > 
> Creates a substring from index 1 up to (not including) the final index, effectively removing the first and last letters from 'oxygen'
> > 
> > 
> >
> > How can we rewrite the slice for getting the last three characters of `element`, so that it works even if we assign a different string to `element`? Test your solution with the following strings: `service`, `clone`, `hi`.
> > 
> > 
> > ## Solution
> > 
> > ```python
> > element = 'oxygen'
> > print('last three characters:', element[-3:])
> > element = 'service'
> > print('last three characters:', element[-3:])
> > element = 'clone'
> > print('last three characters:', element[-3:])
> > element = 'hi'
> > print('last three characters:', element[-3:])
> > ```
> >
> > ```output
> > last three characters: gen
> > last three characters: ice
> > last three characters: one
> > last three characters: hi
> ```
> {: .solution}
{: .challenge}


> ## Thin Slices
> 
> The expression `element[3:3]` produces an [empty string](../learners/reference.md#empty-string),
> i.e., a string that contains no characters. If `data` holds our array of patient data,
> what does `data[3:3, 4:4]` produce? What about `data[3:3, :]`?
> 
> > ## Solution
> > 
> > ```output
> > array([], shape=(0, 0), dtype=float64)
> > array([], shape=(0, 40), dtype=float64)
> ```
> {: .solution}
{: .challenge}


> ## Stacking Arrays
> 
> Arrays can be concatenated and stacked on top of one another, using NumPy's `vstack` and `hstack` functions for vertical and horizontal stacking, respectively.
> 
> ~~~
> import numpy as np
> 
> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
> print('A = ')
> print(A)
> 
> B = np.hstack([A, A])
> print('B = ')
> print(B)
> 
> C = np.vstack([A, A])
> print('C = ')
> print(C)
> ~~~
> {: .python}
> 
> ```output
> A =
> [[1 2 3]
> [4 5 6]
> [7 8 9]]
> B =
> [[1 2 3 1 2 3]
>  [4 5 6 4 5 6]
>  [7 8 9 7 8 9]]
> C =
> [[1 2 3]
> [4 5 6]
> [7 8 9]
> [1 2 3]
> [4 5 6]
> [7 8 9]]
>```
> 
> Write some additional code that slices the first and last columns of `A`, and stacks them into a 3x2 array. Make sure to `print` the results to verify your solution.
> 
> 
> > ## Solution
> > 
> > A 'gotcha' with array indexing is that singleton dimensions are dropped by default. That means `A[:, 0]` is a one dimensional array, which won't stack as desired. To preserve singleton dimensions,the index itself can be a slice or array. For example, `A[:, :1]` returns a two dimensional array with one singleton dimension (i.e. a column vector).
> >
> > ~~~
> > D = np.hstack((A[:, :1], A[:, -1:]))
> > print('D = ')
> > print(D)
> > ~~~
> > {: .python}
> > 
> > ```output
> > D =
> > [[1 3]
> >  [4 6]
> >  [7 9]]
> > ```
> {: .solution}
> 
> > ## Solution
> > 
> > An alternative way to achieve the same result is to use Numpy's delete function to remove the second column of A. If you're not sure what the parameters of numpy.delete mean, use the help files.
> > 
> > ~~~
> > D = np.delete(arr=A, obj=1, axis=1)
> > print('D = ')
> > print(D)
> > ~~~
> > {: .python}
> > 
> > ```output
> > D =
> > [[1 3]
> > [4 6]
> > [7 9]]
> > ```
> {: .solution}
> 
> ## Change In Inflammation
> 
> The patient data is *longitudinal* in the sense that each row represents a series of observations relating to one individual.  This means that the change in inflammation over time is a meaningful concept. Let's find out how to calculate changes in the data contained in an array with NumPy.
> 
> The `np.diff()` function takes an array and returns the differences between two successive values. Let's use it to examine the changes each day across the first week of patient 3 from our inflammation dataset.
> 
> ~~~
> patient3_week1 = data[3, :7]
> print(patient3_week1)
> ~~~
> {: .python}
> 
> ```output
>  [0. 0. 2. 0. 4. 2. 2.]
> ```
> 
> Calling `np.diff(patient3_week1)` would do the following calculations
> 
> ```python
> [ 0 - 0, 2 - 0, 0 - 2, 4 - 0, 2 - 4, 2 - 2 ]
> ```
> 
> and return the 6 difference values in a new array.
> 
> ```python
> np.diff(patient3_week1)
> ```
> 
> ```output
> array([ 0.,  2., -2.,  4., -2.,  0.])
> ```
> 
> Note that the array of differences is shorter by one element (length 6).
> 
> When calling `np.diff` with a multi-dimensional array, an `axis` argument may be passed to the function to specify which axis to process. When applying `np.diff` to our 2D inflammation array `data`, which axis would we specify?
> 
> 
> > ## Solution
> > 
> > Since the row axis (0) is patients, it does not make sense to get the difference between two arbitrary patients. The column axis (1) is in days, so the difference is the change in inflammation -- a meaningful concept.
> >  
> > ~~~
> >  np.diff(data, axis=1)
> > ~~~
> > {: .python}
> {: .solution}
> 
> If the shape of an individual data file is `(60, 40)` (60 rows and 40 columns), what would the shape of the array be after you run the `diff()` function and why?
> 
> 
> > ## Solution
> > 
> > The shape will be `(60, 39)` because there is one fewer difference between columns than there are columns in the data.
> {: .solution}
> 
> How would you find the largest change in inflammation for each patient? Does it matter if the change in inflammation is an increase or a decrease?
> 
> > ## Solution
> > 
> > By using the `np.amax()` function after you apply the `np.diff()` function, you will get the largest difference between days.
> > 
> > ~~~
> > np.amax(np.diff(data, axis=1), axis=1)
> > ~~~
> > {: .python}
> > 
> > ~~~
> > array([  7.,  12.,  11.,  10.,  11.,  13.,  10.,   8.,  10.,  10.,   7.,
> >          7.,  13.,   7.,  10.,  10.,   8.,  10.,   9.,  10.,  13.,   7.,
> >         12.,   9.,  12.,  11.,  10.,  10.,   7.,  10.,  11.,  10.,   8.,
> >         11.,  12.,  10.,   9.,  10.,  13.,  10.,   7.,   7.,  10.,  13.,
> >         12.,   8.,   8.,  10.,  10.,   9.,   8.,  13.,  10.,   7.,  10.,
> >          8.,  12.,  10.,   7.,  12.])
> > ~~~
> > {: .python}
> > 
> > If inflammation values *decrease* along an axis, then the difference from one element to the next will be negative. If you are interested in the **magnitude** of the change and not the direction, the `np.absolute()` function will provide that.
> > 
> > Notice the difference if you get the largest *absolute* difference between readings.
> > 
> > ~~~
> > np.amax(np.absolute(np.diff(data, axis=1)), axis=1)
> > ~~~
> > {: .python}
> > 
> > ~~~
> > array([ 12.,  14.,  11.,  13.,  11.,  13.,  10.,  12.,  10.,  10.,  10.,
> >         12.,  13.,  10.,  11.,  10.,  12.,  13.,   9.,  10.,  13.,   9.,
> >         12.,   9.,  12.,  11.,  10.,  13.,   9.,  13.,  11.,  11.,   8.,
> >         11.,  12.,  13.,   9.,  10.,  13.,  11.,  11.,  13.,  11.,  13.,
> >         13.,  10.,   9.,  10.,  10.,   9.,   9.,  13.,  10.,   9.,  10.,
> >         11.,  13.,  10.,  10.,  12.])
> > ~~~
> > {: .python}
> {: .solution}
{: .challenge}

## Analyzing multiple data


As a final piece to processing our inflammation data, we need a way to get a list of all the files
in our `data` directory whose names start with `inflammation-` and end with `.csv`.
The following library will help us to achieve this:

```python
from glob import glob
```

The `glob` library contains a function, also called `glob`,
that finds files and directories whose names match a pattern.
We provide those patterns as strings:
the character `*` matches zero or more characters,
while `?` matches any one character.
We can use this to get the names of all the CSV files in the current directory:

```python
print(glob('data/inflammation*.csv'))
```

```output
['inflammation-05.csv', 'inflammation-11.csv', 'inflammation-12.csv', 'inflammation-08.csv',
'inflammation-03.csv', 'inflammation-06.csv', 'inflammation-09.csv', 'inflammation-07.csv',
'inflammation-10.csv', 'inflammation-02.csv', 'inflammation-04.csv', 'inflammation-01.csv']
```

As these examples show,
`glob`'s result is a list of file and directory paths in arbitrary order.
This means we can loop over it
to do something with each filename in turn.
In our case,
the "something" we want to do is generate a set of plots for each file in our inflammation dataset.

If we want to start by analyzing just the first three files in alphabetical order, we can use the
`sorted` built-in function to generate a new sorted list from the `glob` output:

~~~
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

filenames = sorted(glob('data/inflammation*.csv'))
filenames = filenames[0:3]
for filename in filenames:
    print(filename)

    data = np.loadtxt(fname=filename, delimiter=',')

    fig = plt.figure(figsize=(10.0, 3.0))

    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)

    axes1.set_ylabel('average')
    axes1.plot(np.mean(data, axis=0))

    axes2.set_ylabel('max')
    axes2.plot(np.amax(data, axis=0))

    axes3.set_ylabel('min')
    axes3.plot(np.amin(data, axis=0))

    fig.tight_layout()
    plt.show()
~~~
{: .python}

```output
inflammation-01.csv
```

![Output from the first iteration of the for loop. Three line graphs showing the daily average, maximum and minimum inflammation over a 40-day period for all patients in the first dataset.](../fig/03-loop_49_1.png)

```output
inflammation-02.csv
```

![Output from the second iteration of the for loop. Three line graphs showing the daily average, maximum and minimum inflammation over a 40-day period for all patients in the seconddataset.](../fig/03-loop_49_3.png)

```output
inflammation-03.csv
```

![Output from the third iteration of the for loop. Three line graphs showing the daily average, maximum and minimum inflammation over a 40-day period for all patients in the thirddataset.](../fig/03-loop_49_5.png)

The plots generated for the second clinical trial file look very similar to the plots for
the first file: their average plots show similar "noisy" rises and falls; their maxima plots
show exactly the same linear rise and fall; and their minima plots show similar staircase
structures.

The third dataset shows much noisier average and maxima plots that are far less suspicious than
the first two datasets, however the minima plot shows that the third dataset minima is
consistently zero across every day of the trial. If we produce a heat map for the third data file
we see the following:

![Heat map of the third inflammation dataset. Note that there are sporadic zero values throughoutthe entire dataset, and the last patient only has zero values over the 40 day study.](../fig/inflammation-03-imshow.svg)

We can see that there are zero values sporadically distributed across all patients and days of the
clinical trial, suggesting that there were potential issues with data collection throughout the
trial. In addition, we can see that the last patient in the study didn't have any inflammation
flare-ups at all throughout the trial, suggesting that they may not even suffer from arthritis!


> ## Plotting Differences
> 
> Plot the difference between the average inflammations reported in the first and second datasets
> (stored in `inflammation-01.csv` and `inflammation-02.csv`, correspondingly), i.e., the difference between the leftmost plots of the first two figures.
> 
> > ## Solution
> > 
> > ~~~
> > from glob import glob
> > import numpy
> > import matplotlib.pyplot as plt
> > 
> > filenames = sorted(glob('data/inflammation*.csv'))
> > 
> > data0 = np.loadtxt(fname=filenames[0], delimiter=',')
> > data1 = np.loadtxt(fname=filenames[1], delimiter=',')
> > 
> > fig = plt.figure(figsize=(10.0, 3.0))
> > 
> > plt.ylabel('Difference in average')
> > plt.plot(np.mean(data0, axis=0) - np.mean(data1, axis=0))
> > 
> > fig.tight_layout()
> > plt.show()
> > ~~~
> > {: .python}
> {: .solution}
{: .challenge}



> ## Generate Composite Statistics
> 
> Use each of the files once to generate a dataset containing values averaged over all patients by completing the code inside the loop given below:
> 
> ~~~
> filenames = glob('data/inflammation*.csv')
> composite_data = numpy.zeros((60, 40))
> for filename in filenames:
>    # sum each new file's data into composite_data as it's read
>    #
> # and then divide the composite_data by number of samples
> composite_data = composite_data / len(filenames)
> ~~~
> {: .python}
> 
> Then use pyplot to generate average, max, and min for all patients.
> 
> > ## Solution
> > 
> > ~~~
> > from glob import glob
> > import numpy as np
> > import matplotlib.pyplot as plt
> > 
> > filenames = glob('data/inflammation*.csv')
> > composite_data = np.zeros((60, 40))
> > 
> > for filename in filenames:
> >     data = np.loadtxt(fname = filename, delimiter=',')
> >     composite_data = composite_data + data
> > 
> > composite_data = composite_data / len(filenames)
> > 
> > fig = plt.figure(figsize=(10.0, 3.0))
> > 
> > axes1 = fig.add_subplot(1, 3, 1)
> > axes2 = fig.add_subplot(1, 3, 2)
> > axes3 = fig.add_subplot(1, 3, 3)
> > 
> > axes1.set_ylabel('average')
> > axes1.plot(np.mean(composite_data, axis=0))
> > 
> > axes2.set_ylabel('max')
> > axes2.plot(np.amax(composite_data, axis=0))
> > 
> > axes3.set_ylabel('min')
> > axes3.plot(np.amin(composite_data, axis=0))
> > 
> > fig.tight_layout()
> > 
> > plt.show()
> > ~~~
> > {: .python}
> {: .solution}
{: .chellenge}

After spending some time investigating the heat map and statistical plots, as well as
doing the above exercises to plot differences between datasets and to generate composite
patient statistics, we gain some insight into the twelve clinical trial datasets.

The datasets appear to fall into two categories:

- seemingly "ideal" datasets that agree excellently with Dr. Maverick's claims,
  but display suspicious maxima and minima (such as `inflammation-01.csv` and `inflammation-02.csv`)
- "noisy" datasets that somewhat agree with Dr. Maverick's claims, but show concerning
  data collection issues such as sporadic missing values and even an unsuitable candidate
  making it into the clinical trial.

In fact, it appears that all three of the "noisy" datasets (`inflammation-03.csv`,
`inflammation-08.csv`, and `inflammation-11.csv`) are identical down to the last value.
Armed with this information, we confront Dr. Maverick about the suspicious data and
duplicated files.

Dr. Maverick has admitted to fabricating the clinical data for their drug trial. They did this after discovering that the initial trial had several issues, including unreliable data recording and poor participant selection. In order to prove the efficacy of their drug, they created fake data. When asked for additional data, they attempted to generate more fake datasets, and also included the original poor-quality dataset several times in order to make the trials seem more realistic.

Congratulations! We've investigated the inflammation data and proven that the datasets have been
synthetically generated.

But it would be a shame to throw away the synthetic datasets that have taught us so much
already, so we'll forgive the imaginary Dr. Maverick and continue to use the data to learn
how to program.






