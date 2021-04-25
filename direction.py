
class direction():
   Up = False
   Right = False
   Left = False
   Down = False

   def __init__(self, Up = None, Right = None,Down = None,Left = None):
      self.Up = Up if Up is not None else None
      self.Right = Right if Right is not None else None
      self.Left = Left if Left is not None else None
      self.Down = Down if Down is not None else None

   def __str__(self):
      # Do whatever you want here
      return "Up: {0} Right: {1} Down: {2} Left: {3} ".format(self.Up, self.Right, self.Down, self.Left)
