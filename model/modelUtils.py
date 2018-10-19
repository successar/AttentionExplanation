def isTrue(obj, attr) :
    return hasattr(obj, attr) and getattr(obj, attr)