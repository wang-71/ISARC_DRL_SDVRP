import torch

# ====== Default Parameters ======
DEFAULT_EQUAL_DEMAND   = True          # equal demand for every node
DEFAULT_MIN_DEMAND     = 5
DEFAULT_MAX_DEMAND     = 11
DEFAULT_SPECIAL_IDS    = [1, 9, 10, 18, 5, 14]  # corner and middle
#DEFAULT_SPECIAL_IDS    = [1, 9, 10, 18] # corner only
DEFAULT_SPECIAL_FACTOR = 2.0
DEFAULT_IDS_ONE_BASED  = True
# ==========================================

# ====== Default Capacities======
DEFAULT_CAPACITIES = {
    10: 20.0,
    15: 25.0,
    18: 100.0,   # <<< Application in ISARC example
    20: 30.0,
    30: 33.0,
    40: 37.0,
    50: 40.0,
    60: 43.0,
    75: 45.0,
    98: 125.0,
    100: 50.0,
    125: 55.0,
    150: 60.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}

def set_capacity_for(num_loc: int, value: float):
    DEFAULT_CAPACITIES[int(num_loc)] = float(value)
# ==========================================

class RectSampler:
    """Get data within [x0,x1]×[y0,y1] region"""
    def __init__(self, x0, x1, y0, y1, device=None, dtype=torch.float):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.device, self.dtype = device, dtype

    def sample(self, shape):
        assert shape[-1] == 2
        u = torch.rand(*shape[:-1], 2, device=self.device, dtype=self.dtype)
        xs = self.x0 + (self.x1 - self.x0) * u[..., 0]
        ys = self.y0 + (self.y1 - self.y0) * u[..., 1]
        return torch.stack([xs, ys], dim=-1)


class OuterMinusInnerSampler:
    """Get data within outer region but outside inner region (storage)"""
    def __init__(self, outer, inner, device=None, dtype=torch.float):
        self.outer = outer
        self.inner = inner
        self.outer_sampler = RectSampler(*outer, device=device, dtype=dtype)

    def _inside_inner(self, pts):
        x0i, x1i, y0i, y1i = self.inner
        return (pts[..., 0] >= x0i) & (pts[..., 0] <= x1i) & \
               (pts[..., 1] >= y0i) & (pts[..., 1] <= y1i)

    def sample(self, shape):
        assert shape[-1] == 2
        pts = self.outer_sampler.sample(shape)
        mask = self._inside_inner(pts)
        loop_guard = 0
        while mask.any() and loop_guard < 20:
            new_pts = self.outer_sampler.sample((mask.sum(), 2))
            pts[mask] = new_pts
            mask = self._inside_inner(pts)
            loop_guard += 1
        return pts


class RectRowsAlignedSampler:
    """
    Generate aligned datasets within [x0i,x1i]×[y0i,y1i]：
    - Number of columns K = N//2
    - Make sure x coordinate of the same column align
    - y = cy ± dy/2，dy ∈ [vmin, vmax]，and dy < H 
    - dx ∈ [hmin, hmax]，make sure K columns are within W
    """
    def __init__(
        self,
        x0i, x1i, y0i, y1i,
        vgap_range,
        hgap_range,
        allow_odd: bool = False,
        device=None, dtype=torch.float
    ):
        self.x0i, self.x1i, self.y0i, self.y1i = x0i, x1i, y0i, y1i
        self.vmin, self.vmax = vgap_range
        self.hmin, self.hmax = hgap_range
        self.allow_odd = allow_odd
        self.device, self.dtype = device, dtype

    def sample(self, shape):
        assert shape[-1] == 2
        N = int(shape[-2])
        batch_shape = shape[:-2]  # e.g. (B,) or (B1,B2,...)

        W = self.x1i - self.x0i
        H = self.y1i - self.y0i
        cx = (self.x0i + self.x1i) / 2.0
        cy = (self.y0i + self.y1i) / 2.0

        if N % 2 == 1 and not self.allow_odd:
            raise ValueError
        K = N // 2 if N % 2 == 0 else N // 2

        eps = torch.tensor(1e-6, device=self.device, dtype=self.dtype)

        # collect dy in every single instance: shape = (*batch,)
        dy_raw = torch.empty(batch_shape, device=self.device, dtype=self.dtype).uniform_(self.vmin, self.vmax)
        dy = torch.minimum(dy_raw, (H - 2 * eps))
        y_top = cy + dy / 2.0  # (*batch,)
        y_bot = cy - dy / 2.0

        if K <= 1:
            xs = torch.full(batch_shape + (1,), cx, device=self.device, dtype=self.dtype)  # (*batch,1)
        else:
            # collect dx in every single instance：shape = (*batch,)
            dx_raw = torch.empty(batch_shape, device=self.device, dtype=self.dtype).uniform_(self.hmin, self.hmax)
            dx_max = torch.tensor(W / (K - 1), device=self.device, dtype=self.dtype)
            dx = torch.minimum(dx_raw, dx_max)  # (*batch,)

            total_span = dx * (K - 1)               # (*batch,)
            x_start = cx - total_span / 2.0         # (*batch,)

            ar = torch.arange(K, device=self.device, dtype=self.dtype)  # (K,)
            xs = x_start[..., None] + dx[..., None] * ar[None, ...]     # (*batch, K)
            xs = torch.clamp(xs, self.x0i + eps, self.x1i - eps)

        # top/bot: (*batch, K, 2)
        top_pts = torch.stack([xs, y_top[..., None].expand_as(xs)], dim=-1)
        bot_pts = torch.stack([xs, y_bot[..., None].expand_as(xs)], dim=-1)

        pts = torch.cat([top_pts, bot_pts], dim=-2)  # (*batch, 2K, 2)

        if N % 2 == 1 and self.allow_odd:
            extra = torch.stack([
                torch.full(batch_shape, cx, device=self.device, dtype=self.dtype),
                y_top
            ], dim=-1)[..., None, :]                   # (*batch,1,2)
            pts = torch.cat([pts, extra], dim=-2)      # (*batch, 2K+1, 2)

        return pts


class DemandSamplerWithIDs:

    def __init__(
        self,
        min_demand=3, max_demand=7,            
        equal=False,
        special_ids=None,
        factor=2.0,
        one_based=True,
        device=None, dtype=torch.float
    ):
        self.min_demand = int(min_demand)
        self.max_demand = int(max_demand)
        self.equal = bool(equal)
        self.factor = float(factor)
        self.one_based = bool(one_based)
        self.device, self.dtype = device, dtype

        self.special_ids = [] if special_ids is None else list(special_ids)

    def sample(self, shape):
        # shape = (*batch, num_loc)
        assert len(shape) >= 1
        num_loc = shape[-1]
        batch_shape = tuple(shape[:-1]) if len(shape) > 1 else (1,)

        if self.equal:
            base = torch.randint(
                low=self.min_demand - 1,
                high=self.max_demand,
                size=batch_shape + (1,),
                device=self.device,
                dtype=torch.long,
            ).to(self.dtype)
            k = base.expand(batch_shape + (num_loc,))
        else:
            k = torch.randint(
                low=self.min_demand - 1,
                high=self.max_demand,
                size=batch_shape + (num_loc,),
                device=self.device,
                dtype=torch.long,
            ).to(self.dtype)

        if len(self.special_ids) > 0:
            idx0 = torch.tensor(
                [(i - 1 if self.one_based else i) for i in self.special_ids],
                device=self.device, dtype=torch.long
            )
            idx0 = idx0[(idx0 >= 0) & (idx0 < num_loc)]
            if idx0.numel() > 0:
                mask = torch.zeros(batch_shape + (num_loc,), device=self.device, dtype=torch.bool)
                mask[..., idx0] = True

                tgt = torch.floor(self.factor * (k + 1.0))
                k = torch.where(mask, tgt - 1.0, k)

        return k 
    
def make_generator_params(
    num_loc,
    outer_size=(8, 6),
    inner_size=(4, 2),
    vgap_range=(1.0, 2.0),
    hgap_range=(0.3, 0.5),
    allow_odd=False,
    equal_demand=DEFAULT_EQUAL_DEMAND,
    capacity: float | None = None,
):
    W, H = outer_size
    w, h = inner_size

    cx, cy = W / 2, H / 2
    x0i, x1i = cx - w / 2, cx + w / 2
    y0i, y1i = cy - h / 2, cy + h / 2
    x0o, x1o, y0o, y1o = 0.0, W, 0.0, H

    loc_sampler = RectRowsAlignedSampler(
        x0i, x1i, y0i, y1i,
        vgap_range=vgap_range,
        hgap_range=hgap_range,
        allow_odd=allow_odd,
    )

    depot_sampler = OuterMinusInnerSampler(
        outer=(x0o, x1o, y0o, y1o),
        inner=(x0i, x1i, y0i, y1i)
    )

    cap = capacity if capacity is not None else DEFAULT_CAPACITIES.get(int(num_loc), None)

    params = dict(
        num_loc=num_loc,
        min_loc=0.0,
        max_loc=max(W, H),
        loc_sampler=loc_sampler,
        depot_sampler=depot_sampler,
        min_demand=DEFAULT_MIN_DEMAND,
        max_demand=DEFAULT_MAX_DEMAND,
        capacity=cap,  
    )

    # Special Nodes with Strengthened Connection
    params["demand_sampler"] = DemandSamplerWithIDs(
        min_demand=DEFAULT_MIN_DEMAND,
        max_demand=DEFAULT_MAX_DEMAND,
        equal=equal_demand,
        special_ids=DEFAULT_SPECIAL_IDS,
        factor=DEFAULT_SPECIAL_FACTOR,
        one_based=DEFAULT_IDS_ONE_BASED,
    )
    
    params["outer_size"] = outer_size 

    return params
