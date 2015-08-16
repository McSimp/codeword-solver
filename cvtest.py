from cvstuff import CrosswordRecogniserInterface

cri = CrosswordRecogniserInterface()
structure = cri.solve_image('cvstuff/test1.jpg')

for row in structure:
    print row

cri.exit()
