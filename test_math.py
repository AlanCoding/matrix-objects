from math_funcs import *
from nose.tools import raises


def is_equal(x, y, tolerance=0.001):
    """Helper function to compare floats, which are often not quite equal
    even when they should be."""
    return abs(x - y) <= tolerance


m = Vector([3, 4])
n = Vector([5, 0])

v = Vector([1, 3, 0])
w = Vector([0, 2, 4])
u = Vector([1, 1, 1])
y = Vector([10, 20, 30])
z = Vector([0, 0, 0])


def test_shape_vectors():
    """shape should take a vector or matrix and return a tuple with the
    number of rows (for a vector) or the number of rows and columns
    (for a matrix.)"""
    assert m.shape() == (2,)
    assert v.shape() == (3,)
    assert Vector([1]).shape() == (1,)


def test_vector_add():
    """
    [a b]  + [c d]  = [a+c b+d]

    Matrix + Matrix = Matrix
    """
    assert v+w == Vector([1,5,4])
    assert u+y == Vector([11,21,31])
    assert u+z == u


def test_vector_add_is_communicative():
    assert w+y == y+w


@raises(ShapeException)
def test_vector_add_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    m+v


def test_vector_sub():
    """
    [a b]  - [c d]  = [a-c b-d]

    Matrix + Matrix = Matrix
    """
    assert v-w == Vector([1,1,-4])
    assert w-v == Vector([-1,-1,4])
    assert y-z == y


@raises(ShapeException)
def test_vector_sub_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    m-v


def test_vector_sum():
    """vector_sum can take any number of vectors and add them together."""
    assert v+w+u+y+z == Vector([12,26,35])


@raises(ShapeException)
def test_vector_sum_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    v+w+m+y


def test_dot():
    """
    dot([a b], [c d])   = a * c + b * d

    dot(Vector, Vector) = Scalar
    """
    w.dot(y) == 160
    m.dot(n) == 15
    u.dot(z) == 0


@raises(ShapeException)
def test_dot_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    v.dot(m)


def test_vector_multiply():
    """
    [a b]  *  Z     = [a*Z b*Z]

    Vector * Scalar = Vector
    """
    assert v*0.5 == Vector([0.5,1.5,0])
    assert m*2 == Vector([6,8])


def test_magnitude():
    """
    magnitude([a b])  = sqrt(a^2 + b^2)

    magnitude(Vector) = Scalar
    """
    assert m.magnitude() ==5
    assert y.magnitude() == math.sqrt(1400)



A = Matrix([[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]])
B = Matrix([[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]])
C = Matrix([[1, 2],
     [2, 1],
     [1, 2]])
D = Matrix([[1, 2, 3],
     [3, 2, 1]])


def test_shape_matrices():
    """shape should take a vector or matrix and return a tuple with the
    number of rows (for a vector) or the number of rows and columns
    (for a matrix.)"""
    assert A.shape() == (3,3)
    assert C.shape() == (3,2)
    assert D.shape() == (2,3)


def test_matrix_row():
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    assert A.row(0) == Vector([1,0,0])
    assert B.row(1) == Vector([4,5,6])
    assert C.row(2) == Vector([1,2])


def test_matrix_col():
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    assert A.col(0) == Vector([1,0,0])
    assert B.col(1) == Vector([2,5,8])
    assert D.col(2) == Vector([3,1])


def test_matrix_scalar_multiply():
    """
    [[a b]   *  Z   =   [[a*Z b*Z]
     [c d]]              [c*Z d*Z]]

    Matrix * Scalar = Matrix
    """
    assert C*3 == Matrix([[3,6],
                            [6,3],
                            [3,6]])


def test_matrix_vector_multiply():
    """
    [[a b]   *  [x   =   [a*x+b*y
     [c d]       y]       c*x+d*y
     [e f]                e*x+f*y]

    Matrix * Vector = Vector
    """
    assert A*Vector([2,5,4]) == Vector([2,5,4])


@raises(ShapeException)
def test_matrix_vector_multiply_checks_shapes():
    """Shape Rule: The number of rows of the vector must equal the number of
    columns of the matrix."""
    C*Vector([1,2,3])


def test_matrix_matrix_multiply():
    """
    [[a b]   *  [[w x]   =   [[a*w+b*y a*x+b*z]
     [c d]       [y z]]       [c*w+d*y c*x+d*z]
     [e f]                    [e*w+f*y e*x+f*z]]

    Matrix * Matrix = Matrix
    """
    assert A*B == B
    assert B*C == Matrix([[8, 10],
                        [20, 25],
                        [32, 40]])


@raises(ShapeException)
def test_matrix_matrix_multiply_checks_shapes():
    """Shape Rule: The number of columns of the first matrix must equal the
    number of rows of the second matrix."""
    A*D

def test_identity_matrix():
    A = Matrix([[0]])
    def f(i,j):
        if i==j:
            return 1
        else:
            return 0
    A.make_by_func(2,2,f)
    assert A == Matrix([[1,0],[0,1]])
