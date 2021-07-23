
def outputMessage(func):  # デコレータ
    def wrapper(a):

        print("start!!")


        result = func(a)  # これはtest()
        print("end!!")
        return result
    return wrapper


@outputMessage
def test(a):
    print("Hello World", a)
    return "OK"
if __name__ == "__main__":
    print(test("What"))
