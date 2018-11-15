class CatFaceLandmark:
    LEFT_EYE = 0
    RIGHT_EYE = 1
    MOUTH = 2
    LEFT_EAR1 = 3
    LEFT_EAR2 = 4
    LEFT_EAR3 = 5
    RIGHT_EAR1 = 6
    RIGHT_EAR2 = 7
    RIGHT_EAR3 = 8

    def __init__(self):
        pass

    @staticmethod
    def all():
        return [
            {
                'value': CatFaceLandmark.LEFT_EYE,
                'name': 'Left Eye'
            },
            {
                'value': CatFaceLandmark.RIGHT_EYE,
                'name': 'Right Eye'
            },
            {
                'value': CatFaceLandmark.MOUTH,
                'name': 'Mouth'
            },
            {
                'value': CatFaceLandmark.LEFT_EAR1,
                'name': 'Left Ear-1'
            },
            {
                'value': CatFaceLandmark.LEFT_EAR2,
                'name': 'Left Ear-2'
            },
            {
                'value': CatFaceLandmark.LEFT_EAR3,
                'name': 'Left Ear-3'
            },
            {
                'value': CatFaceLandmark.RIGHT_EAR1,
                'name': 'Right Ear-1'
            },
            {
                'value': CatFaceLandmark.RIGHT_EAR2,
                'name': 'Right Ear-2'
            },
            {
                'value': CatFaceLandmark.RIGHT_EAR3,
                'name': 'Right Ear-3'
            }
        ]
