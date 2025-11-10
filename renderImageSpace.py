import torch
import numpy as np

# small helper: polar -> cartesian (z=0)
def polar_to_cartesian(angle_deg, distance):
    rad = np.deg2rad(angle_deg)
    x = distance * np.cos(rad)
    y = distance * np.sin(rad)
    return x, y

def make_scene_from_splats(splats, device, dtype=torch.float32, sigma_min=1e-2,
                           assume_scales_logspace=False, ensure_quats=True):
    """
    Create scene_dict_all from a list of splats.

    Args:
        splats: list of dicts with keys
            - angle (degrees) or 'pos' (tuple x,y)
            - distance (if using angle)
            - sigma (scalar or tuple/list (sx,sy))  in world units
            - color (3-tuple 0..1)
            - opacity (0..1)
        device: torch device
        dtype: torch dtype
        sigma_min: minimum allowed sigma per-axis
        assume_scales_logspace: if True, we store log(scales) to match models that expect `exp(...)` later
        ensure_quats: add identity quats if none provided

    Returns:
        scene_dict_all: dict with tensors means, quats, opacities, scales, colors (on device)
    """

    means_list = []
    quats_list = []
    colors_list = []
    opac_list = []
    scales_list = []

    for s in splats:
        if 'pos' in s:
            x,y = s['pos'][:2]
        else:
            x,y = polar_to_cartesian(s['angle'], s.get('distance', 1.0))
        z = s.get('z', 0.0)  # allow overriding; default z=0

        means_list.append([x, y, z])

        # quaternion: identity if not specified (x,y,z,w) or (w,x,y,z) depending on your code; 
        # Nerfstudio usually uses (x,y,z,w) order in tensors; your loaded quats likely reflect that.
        if 'quat' in s:
            quats_list.append(s['quat'])
        else:
            # identity quaternion (x=0,y=0,z=0,w=1)
            quats_list.append([0.0, 0.0, 0.0, 1.0])

        color = s.get('color', (1.0, 1.0, 1.0))
        colors_list.append(list(color))

        opacity = float(s.get('opacity', 1.0))
        opac_list.append(opacity)

        sigma = s.get('sigma', 0.5)
        if isinstance(sigma, (float, int)):
            sx = sy = max(sigma, sigma_min)
            sz = max(s.get('sigma_z', 1e-3), 1e-6)
        else:
            sx = max(sigma[0], sigma_min)
            sy = max(sigma[1], sigma_min)
            sz = max(sigma[2] if len(sigma)>2 else 1e-3, 1e-6)
        # scales vector per-axis
        scales_list.append([sx, sy, sz])

    N = len(means_list)
    means = torch.tensor(means_list, dtype=dtype, device=device).view(N,3)

    quats = torch.tensor(quats_list, dtype=dtype, device=device).view(N,4)

    colors = torch.tensor(colors_list, dtype=dtype, device=device).view(N,3)

    opacities = torch.tensor(opac_list, dtype=dtype, device=device).view(N)  # [N]

    scales = torch.tensor(scales_list, dtype=dtype, device=device).view(N,3)  # per-axis scales

    if assume_scales_logspace:
        # if model expects params in log-space (and applies exp later), store logs
        scales_param = torch.log(torch.clamp(scales, min=1e-6))
    else:
        scales_param = scales  # pass raw positive scales

    scene_dict_all = {
        'means': means,
        'quats': quats,
        'opacities': opacities,
        'scales': scales_param,
        'colors': colors
    }
    return scene_dict_all


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test code
my_splats = []
for angle in [0, 45, 90, 180, 270]:
    my_splats.append({
        'angle': angle,
        'distance': 3.0,       # meters (world units)
        'sigma': 0.2,          # spread in world units
        'color': (np.random.rand(), np.random.rand(), np.random.rand()),
        'opacity': 0.8
    })

scene_dict_all = make_scene_from_splats(my_splats, device=DEVICE, assume_scales_logspace=False)
gauss_num = scene_dict_all['means'].size(0)
print(f"Number of Total Gaussians in the Scene: {gauss_num}")
print(f"scene_dict_all {scene_dict_all}")

