# Student Status Database

STUDENT_OK = "STUDENT_OK"
STUDENT_OK_BOARDED_BUS = "STUDENT_OK_BOARDED_BUS"
STUDENT_OK_EXITED_BUS = "STUDENT_OK_EXITED_BUS"
STUDENT_NO_BOARDED_WRONG_BUS = "STUDENT_NO_BOARDED_WRONG_BUS"
STUDENT_NO_DIDNT_BOARD = "STUDENT_NO_DIDNT_BOARD"
STUDENT_NO_EXITED_WRONG_LOCATION = "STUDENT_NO_EXITED_WRONG_LOCATION"
STUDENT_NO_DIDNT_EXIT = "STUDENT_NO_DIDNT_EXIT"

class Student:
    identification = ''
    bus_id = ''
    status = STUDENT_OK
    photo_path = ''

def getInfo (studentName, fieldName):
    if fieldName == "identification":
        return students[studentName].identification
    elif fieldName == "bus_id":
        return students[studentName].bus_id
    elif fieldName == "status":
        return students[studentName].status
    elif fieldName == "photo_path":
        return students[studentName].photo_path
    else:
        print ("Improper input.")

def setInfo (studentName, fieldName, value):
    if fieldName == "identification":
        students[studentName].identification = value
    elif fieldName == "bus_id":
        students[studentName].bus_id = value
    elif fieldName == "status":
        students[studentName].status = value
    elif fieldName == "photo_path":
        students[studentName].photo_path = value
    else:
        print ("Improper input.")

Rene = Student()
Rene.identification = "4462e394-3832-45b5-a8ba-ec39e69fb617"
Rene.bus_id = 512

shari = Student()
shari.identification = "4462e394-3832-45b5-a8ba-ec39e69fb617"
shari.bus_id = 512

mo = Student()
mo.identification = "4462e394-3832-45b5-a8ba-ec39e69fb617"
mo.bus_id = 512



students = {
            "Rene":Rene,
	    "shari":shari,
        "mo":mo}
