import math


class ShapeException(Exception):
    pass


class TypeException(Exception):
    pass


class MatrixVector():

    def __init__(self, some_list):
        self.vals = some_list


    def is_matrix(self):
        v = self.vals
        if type(v[0]) is list:
            return True
        return False

    def shape(self):
        u = self.vals
        if self.is_matrix():
            return (len(u), len(u[0]))
        else:
            return (len(u), )


class Matrix(MatrixVector):

    def __init__(self, A):
        super().__init__(A)
        if self.vals[0] is int:
            for i in range(len(self.vals)):
                self.vals[i] = [self.vals[i]]



    def make_by_func(self, r, c, func):
        self.vals = [[func(j,i) for j in range(c)] for i in range(r)]


    def transpose(self):
        r, c = self.shape()
        newMat = Matrix([[0]])
        def f(i,j):
            return 0
        newMat.make_by_func(c, r, f)
        for i in range(r):
            for j in range(c):
                newMat.vals[i][j] = self.vals[j][i]
        return newMat


    def get_total(self):
        A = self.vals
        r, c = self.shape()
        return sum(sum(A[i][j] for j in range(c)) for i in range(r))

    def __str__(self):
        rtrn = ""
        r, c = self.shape()
        for i in range(r):
            for j in range(c):
                rtrn += str(self.vals[i][j])+" "
            rtrn += " \n"
        return rtrn


    def __eq__(self, other):
        if type(other) is not Matrix:
            return False
        A = self.vals
        B = other.vals
        r, c = self.shape()
        for i in range(r):
            for j in range(c):
                if A[i][j] != B[i][j]:
                    return False
        return True


    def row(self, i):
        A = self.vals
        return Vector([j for j in A[i]])


    def col(self, j):
        A = self.vals
        return Vector([A[i][j] for i in range(len(A))])


    def matrix_scalar_multiply(self, a):
        C = self.vals
        return Matrix([[C[i][j]*a for j in range(len(C[0]))]
                for i in range(len(C))])


    def matrix_matrix_multiply(self, other):
        A = self.vals
        B = other.vals
        if len(A[0]) != len(B):
            raise ShapeException()
        return Matrix([[sum(A[i][k]*B[k][j]
                for k in range(len(A[0])))
                for j in range(len(B[0]))]
                for i in range(len(A))])


    def matrix_vector_multiply(self, other):
        A = self.vals
        v = other.vals
        vMat = Matrix([[val] for val in v])
        matOut = self.matrix_matrix_multiply(vMat)
        C = matOut.vals
        return Vector([C[i][0] for i in range(len(C))])


    def __add__(self, other):
        if type(other) is not Matrix:
            raise TypeException()
        elif self.shape() != other.shape():
            raise ShapeException()
        A = self.vals
        B = other.vals
        r, c = self.shape()
        return Matrix([[A[i][j]+B[i][j] for i in range(r)]
                        for j in range(c)])


    def __sub__(self, other):
        A = self.vals
        nB = other.matrix_scalar_multiply(-1)
        return self.__add__(nB)


    def __rmul__(self, other):
        if type(other) is int or type(other) is float:
            return self.matrix_scalar_multiply(other)
        elif type(other) is Vector or type(other) is Matrix:
            return other.__mul__(self)
        else:
            raise TypeException()


    def __mul__(self,other):
        if type(other) is int or type(other) is float:
            return self.matrix_scalar_multiply(other)
        elif type(other) is Vector:
            return self.matrix_vector_multiply(other)
        elif type(other) is Matrix:
            return self.matrix_matrix_multiply(other)
        else:
            raise TypeException()


class Vector(MatrixVector):


    def __str__(self):
        rtrn = ""
        for i in range(len(self.vals)):
            rtrn += " " + str(self.vals[i])
        return rtrn

    def __add__(self, other):
        u = self.vals
        w = other.vals
        if self.shape() != other.shape():
            raise ShapeException()
        return Vector([i+j for i, j in zip(u, w)])


    def vector_scalar_multiply(self, other):
        return Vector([self.vals[i]*other
                       for i in range(len(self.vals))])

    def __sub__(self, other):
        return self.__add__(other.vector_scalar_multiply(-1))


    def dot(self, other):
        w = self.vals
        y = other.vals
        if self.shape() != other.shape():
            raise ShapeException()
        return sum(i*j for i, j in zip(w, y))


    def vector_matrix_multiply(self, other):
        pass

    def vector_vector_multiply(self, other):
        v = self.vals
        w = other.vals
        return Vector([i*j for i,j in zip(v,w)])



    def __mul__(self, other):
        if type(other) is Vector:
            return self.vector_vector_multiply(other)
        elif type(other) is Matrix:
            return self.vector_matrix_multiply(other)
        elif type(other) is int or type(other) is float:
            return self.vector_scalar_multiply(other)
        else:
            raise TypeException()

    def __rmul__(self,other):
        if type(other) is Matrix:
            return other.matrix_vector_multiply(self)
        elif type(other) is Vector:
            return self.vector_vector_multiply(other)
        elif type(other) is int or type(other) is float:
            return self.vector_scalar_multiply(other)
        else:
            raise TypeException()

    # def vector_mean(*vectors):
    #     theSum = vector_sum(*vectors)
    #     return [i/len(vectors) for i in theSum]

    def magnitude(self):
        v = self.vals
        squares = sum(i**2 for i in v)
        return squares**(0.5)

    def __eq__(self, other):
        if type(other) is not Vector:
            return False
        v = self.vals
        u = other.vals
        return sum(i==j for i,j in zip(v,u))


def drake_pop():

    S = Matrix([[0 , 0.25 , 0.6 , 0.8 , 0.15 , 0 ],
    [0.7 , 0 , 0 , 0 , 0 , 0 ],
    [0 , 0.95 , 0 , 0 , 0 , 0 ],
    [0 , 0 , 0.9 , 0 , 0 , 0 ],
    [0 , 0 , 0 , 0.9 , 0 , 0 ],
    [0 , 0 , 0 , 0 , 0.5 , 0 ]])

    P_st = Matrix([[0]])
    def f(i,j):
        if i==0 and j==0:
            return 10
        else:
            return 0
    P_st.make_by_func(6,6,f)
    print(S*P_st)
    print((S*P_st).get_total())

    def number_in_future(year):
        P = P_st
        for i in range(year):
            P = S*P
        return P.get_total()

    print(number_in_future(1))
    print(number_in_future(20))
    print(number_in_future(40))



if __name__ == '__main__':

    m = [3, 4]
    n = [5, 0]

    v = [1, 3, 0]
    w = [0, 2, 4]
    u = [1, 1, 1]
    y = [10, 20, 30]
    z = [0, 0, 0]


    A = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    B = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    C = [[1, 2],
         [2, 1],
         [1, 2]]
    D = [[1, 2, 3],
         [3, 2, 1]]

    print(Vector(n))
    print(Matrix(A).__str__())
    print("hello world")
    print(Vector(n)+Vector(m))
    print(Vector(n)-Vector(m))
    print(Vector(n)*Vector(m))
    print(Vector(v)*Matrix(A))
    print(' ')
    print(3*Matrix(A))
    print(Matrix(A)*3)
    print(Matrix(A)*5.2)
    print(" ---- class tests -----")
    print(Vector([1, 2]) + Vector([0, 4]))
    print(Vector([1, 2]) - Vector([0, 4]))
    print(Vector([1, 2]) * 3)
    print(" ---- equality -----")
    print(Vector(y)==Vector(z))
    print(Matrix(A)==Matrix(B))
    print(Matrix(A)==Matrix(A))

    print(Vector([1, 2]) == Vector([1, 2])) # results in True

    print(Matrix([[0, 1], [1, 0]]) + Matrix([[1, 1], [0, 0]]))
    print(Matrix([[0, 1], [1, 0]]) - Matrix([[1, 1], [0, 0]]))
    print(Matrix([[0, 1], [1, 0]]) * 3)
    print(" mat-vect multiply ")
    print(Matrix([[0, 1], [1, 0]]) * Vector([1, 2]))
#    print(Matrix([[1, 1, 1], [0, 0, 0]]) * Matrix([[1, 1], [2, 2], [3, 3]]))

    print(Matrix([[0, 1], [1, 0]]) == Matrix([[1, 1], [0, 0]])) # results in False


    A = Matrix([[0]])
    def f(i,j):
        if i==j:
            return 1
        else:
            return 0
    A.make_by_func(10,10,f)
    #print(A)

    print("transpose")
    print(Matrix(B))
    print(Matrix(B).transpose())

    drake_pop()
