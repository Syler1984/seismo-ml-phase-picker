import config.vars as config


class Event:
    """Event class for parsing nordic files"""
    _id = None           # Event id (string)
    _depth = None        # Depth of event epicentre in km (float)
    _magnitude = None    # Event magnitude (float)
    _picks = []          # List of picks (list)

    def __init__(self, id=None, filename=None):
        """Constructor can be supplied with filename for"""
        self._id = id


    def read_from_file(self, filename):
        """Initializes event from nordic file"""
        if len(filename) == 0:
            raise ValueError("filename should not be empty")

        file = open(filename)

        for line in file:


        file.close()

