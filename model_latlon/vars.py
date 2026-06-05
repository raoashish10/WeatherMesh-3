import torch
import numpy as np
from utils import get_date

N_ADDL_VARS = 4

_RES = 0.25
_LONS_1D = np.arange(0, 360, _RES)
_LATS_1D = np.arange(90, -90.01, -_RES)[:-1]
_LONS, _LATS = np.meshgrid(_LONS_1D, _LATS_1D)
_LAT_RAD = np.radians(_LATS)

def _compute_radiation_and_solar(doy, hr):
    """Compute radiation (W/m^2) and solar altitude (degrees) for day-of-year and hour.

    Args:
        doy: day-of-year, 0-based (0 = Jan 1)
        hr: hour of day (0-23)
    Returns:
        (radiation, altitude_deg) arrays of shape (720, 1440), float32
    """
    day = doy + 1

    declination_rad = np.radians(23.45 * np.sin((2 * np.pi / 365.0) * (day - 81)))

    b = 2 * np.pi / 364.0 * (day - 81)
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)

    solar_time = (hr * 60 + 4 * _LONS + eot) / 60.0
    hour_angle_rad = np.radians(15.0 * (solar_time - 12.0))

    sin_alt = np.clip(
        np.cos(_LAT_RAD) * np.cos(declination_rad) * np.cos(hour_angle_rad)
        + np.sin(_LAT_RAD) * np.sin(declination_rad),
        -1, 1
    )
    altitude_deg = np.degrees(np.arcsin(sin_alt))

    is_daytime = altitude_deg > 0
    flux = 1160 + 75 * np.sin(2 * np.pi / 365 * (day - 275))
    optical_depth = 0.174 + 0.035 * np.sin(2 * np.pi / 365 * (day - 100))
    safe_alt_rad = np.where(is_daytime, np.radians(altitude_deg), 1.0)
    radiation = flux * np.exp(-optical_depth / np.sin(safe_alt_rad)) * is_daytime

    return radiation.astype(np.float32), altitude_deg.astype(np.float32)

def get_additional_vars(t0s):
    """
    Adds additional variables to the input tensor which are not dependent on the input data but do change vs time.
    Computes radiation and solar angle on the fly (no external data files needed).

    Args:
        t0s (torch.Tensor): A tensor of shape (B,) containing the time of the input data
    """
    device = t0s.device
    dates = [get_date(t0.item()) for t0 in t0s]
    start_of_years = [date.replace(month=1, day=1) for date in dates]
    time_of_years = [int((date - soy).total_seconds()/86400) for date, soy in zip(dates, start_of_years)]

    rad_list, ang_list = [], []
    for time_of_year, date in zip(time_of_years, dates):
        rad, ang = _compute_radiation_and_solar(time_of_year, date.hour)
        rad_norm = torch.HalfTensor((rad - 300) / 400)
        ang_rad = torch.HalfTensor(np.radians(ang))
        rad_list.append(rad_norm[:720, :, np.newaxis].unsqueeze(0).to(device))
        ang_list.append(ang_rad[:720, :, np.newaxis].unsqueeze(0).to(device))

    radiations = torch.cat(rad_list, dim=0)
    solar_angles = torch.cat(ang_list, dim=0)

    hours = torch.tensor([date.hour/24 for date in dates]).to(device)
    time_of_day = (torch.zeros_like(radiations, device=hours.device) + hours[:,None, None, None])

    sin_angles = torch.sin(solar_angles)
    cos_angles = torch.cos(solar_angles)

    out = torch.cat((radiations, time_of_day, sin_angles, cos_angles), axis=-1)
    assert out.shape[3] == N_ADDL_VARS, f"Expected {N_ADDL_VARS} additional variables, but got {out.shape[3]}"
    return out

def get_constant_vars(mesh):
    const_vars = []
    to_cat = []
        
    latlon = torch.FloatTensor(mesh.xpos)
    slatlon = torch.sin((latlon*torch.Tensor([np.pi/2,np.pi])))
    clatlon = torch.cos((latlon*torch.Tensor([np.pi/2,np.pi])))
    const_vars += ['sinlat','sinlon','coslat','coslon'] 
    to_cat += [slatlon,clatlon]

    land_mask_np = np.load('constants/additional_variables/land_mask.npy')
    land_mask = torch.BoolTensor(np.round(downsample(land_mask_np, mesh.xpos.shape)))
    const_vars += ['land_mask']
    to_cat += [land_mask.unsqueeze(-1)]

    soil_type_np = np.load('constants/additional_variables/soil_type.npy')
    soil_type_np = downsample(soil_type_np, mesh.xpos.shape, reduce=np.min)
    soil_type = torch.BoolTensor(to_onehot(soil_type_np))
    const_vars += [f'soil_type{i}' for i in range(soil_type.shape[-1])]
    to_cat += [soil_type]

    elevation_np = np.load('constants/additional_variables/topography.npy')
    elevation_np = downsample(elevation_np, mesh.xpos.shape, reduce=np.mean)
    elevation_np = elevation_np / np.max(elevation_np)
    elevation = torch.FloatTensor(elevation_np)
    const_vars += ['elevation']
    to_cat += [elevation.unsqueeze(-1)]

    const_data = torch.cat(to_cat, axis=-1)
    
    assert const_data.shape[-1] == len(const_vars), f"{const_data.shape[-1]} vs {len(const_vars)}"
    return const_data, const_vars

def downsample(mask,shape,reduce=np.mean):
    dlat = (mask.shape[0]-1) // shape[0]
    dlon = mask.shape[1] // shape[1]
    assert dlon == dlat
    d = dlat
    toshape = (shape[0], d, shape[1], d)
    ret = reduce(mask[:-1,:].reshape(toshape),axis=(1,3)) # Remove the south pole
    assert ret.shape == shape[:2], (ret.shape, shape[:2])
    return ret

def to_onehot(x):
    x = x.astype(int)
    D = np.max(x)+1
    return np.eye(D)[x]