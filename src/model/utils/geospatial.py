import torch

def latlonh_to_ecef(lat, lon, h):
    """
    Converts lat, lon, ht to ecef coordinates in order
    to compute euclidean distance between two points
    """
    a = 6378137.0
    b = 6356752.314245
    lat_rad = lat * torch.pi / 180.
    lon_rad = lon * torch.pi / 180.
    N_lat = a**2 / torch.sqrt((a**2)*torch.cos(lat_rad)**2 +
                           (b**2)*torch.sin(lat_rad)**2)
    x = (N_lat + h)*torch.cos(lat_rad)*torch.cos(lon_rad)
    y = (N_lat + h)*torch.cos(lat_rad)*torch.sin(lon_rad)
    z = ((b**2)/(a**2)*N_lat + h)*torch.sin(lat_rad)
    return x, y, z

def compute_distance_lonlonh(lat0, lon0, ht0, lat1, lon1, ht1):
    x0, y0, z0 = latlonh_to_ecef(lat0, lon0, ht0)
    x1, y1, z1 = latlonh_to_ecef(lat1, lon1, ht1)
    d = torch.sqrt( (x0 - x1)**2 + (y0 - y1)**2 + (z0 - z1)**2 )
    return d