import numpy as np
import cv2

class PhospheneRenderer:
    def __init__(self, electrode_grid=(32, 32), lattice_structure=(1, 0, 1), 
                 dropout_rate=0.1, intensity_levels=8, 
                 axon_lambda=2.0, temporal_decay=0.2):
        """
        Renderer Biologicamente Plausibile.
        
        Args:
            axon_lambda: Elongation factor (1.0 = circle, >1.0 = streak).
            temporal_decay: How quickly a phosphene fades (0.0 - 1.0).
        """
        self.n_elec_x = electrode_grid[0]
        self.n_elec_y = electrode_grid[1]
        self.levels = intensity_levels
        self.axon_lambda = axon_lambda
        self.temporal_decay = temporal_decay
        
        # 1. Irregular grid (jittering)
        # Surgery is never perfect: add noise to the positions
        base_centers = self._generate_hex_grid(lattice_structure)
        noise = (np.random.rand(*base_centers.shape) - 0.5) * 0.02 
        self.centers = base_centers + noise
        
        # 2. Maschera Dropout
        self.active_mask = np.random.rand(len(self.centers)) > dropout_rate
        
        # 3. Axon map (orientation map)
        # Simulates the direction of nerve fibers towards the optic disc (assumed on the left)
        self.orientations = self._calculate_axon_map()
        
        # 4. Temporal cache (to simulate fading)
        self.prev_activation = np.zeros(len(self.centers))
        
        # Cache Kernels
        self.kernel_cache = {}

    def _generate_hex_grid(self, lattice):
        v1 = lattice[0] / self.n_elec_x
        v2 = lattice[1] / self.n_elec_x
        v3 = lattice[2] / self.n_elec_y
        x_array = np.linspace(v1/2, 1-v1/2, self.n_elec_x)
        x_array_d = np.linspace(v1/2+v2, 1-v1/2+v2, self.n_elec_x)
        y_array = np.linspace(v3/2, 1-v3/2, self.n_elec_y)
        centers = []
        for i in range(self.n_elec_y):
            y_coord = y_array[i]
            curr_x = x_array_d if (i % 2 == 0 and v2 > 0) else x_array
            for x_coord in curr_x:
                centers.append([y_coord, x_coord])
        return np.array(centers)

    def _calculate_axon_map(self):
        """Compute the phosphene rotation angle from its position."""
        angles = []
        # Assume the optic disc is shifted to the left/center
        optic_disc = np.array([0.5, -0.2]) 
        
        for pt in self.centers:
            # Vector towards the optic disc
            dy = pt[0] - optic_disc[0]
            dx = pt[1] - optic_disc[1]
            # The axon runs along this line; the phosphene elongates along this tangent
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        return np.array(angles)

    def _elongated_gaussian(self, size, luminance, sigma, angle, elongation):
        """Generate a rotated, elongated Gaussian (axon streak)."""
        if luminance <= 1e-5: return np.zeros((size, size))
        
        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)
        
        # Rotazione coordinate
        c, s = np.cos(angle), np.sin(angle)
        X_rot = c * X - s * Y
        Y_rot = s * X + c * Y
        
        # Formula Gaussiana Asimmetrica
        # Sigma X is narrow, Sigma Y is elongated (or vice versa depending on elongation)
        sigma_x = sigma
        sigma_y = sigma * elongation
        
        gauss = luminance * np.exp(-((X_rot**2)/(2*sigma_x**2) + (Y_rot**2)/(2*sigma_y**2)))
        return gauss

    def render(self, image):
        if image.dtype == np.uint8:
            image = image.astype(float) / 255.0
            
        h_img, w_img = image.shape
        out_scale = 10
        out_h, out_w = h_img * out_scale, w_img * out_scale
        out_image = np.zeros((out_h, out_w))
        
        rf_size = int(h_img / self.n_elec_y)
        
        for idx, pt in enumerate(self.centers):
            if not self.active_mask[idx]: continue
                
            # 1. Campionamento
            r, c = int(pt[0] * h_img), int(pt[1] * w_img)
            r0, r1 = max(0, r-rf_size), min(h_img, r+rf_size)
            c0, c1 = max(0, c-rf_size), min(w_img, c+rf_size)
            patch = image[r0:r1, c0:c1]
            
            target_lum = np.mean(patch) if patch.size > 0 else 0
            
            # 2. Dinamica Temporale (Simulazione Desensitizzazione)
            # If the stimulus is constant, perception fades
            activation = target_lum
            # Filtro passa-alto temporale semplice:
            # Perception is strong when there is a positive *change* compared to before
            change = activation - self.prev_activation[idx]
            
            # Simplified model: Brightness = Input - Decay * Accumulation
            # Oppure: enfatizzare i transienti (edges in time)
            perceived_lum = activation 
            if self.temporal_decay > 0:
                # If the input is similar to the previous frame, reduce brightness
                perceived_lum = activation * (1.0 - self.temporal_decay * self.prev_activation[idx])
            
            self.prev_activation[idx] = 0.9 * self.prev_activation[idx] + 0.1 * activation
            perceived_lum = np.clip(perceived_lum, 0, 1)

            # Quantizzazione
            if self.levels > 0:
                perceived_lum = np.floor(perceived_lum * self.levels) / self.levels
            
            if perceived_lum <= 0.05: continue

            # 3. Rendering with axons
            angle = self.orientations[idx]
            
            # Cache key: (discrete brightness, discrete angle)
            # Round the angle for caching
            angle_idx = int(np.degrees(angle) / 10) 
            lum_idx = int(perceived_lum * 10)
            cache_key = (lum_idx, angle_idx)
            
            kernel_size = rf_size * out_scale * 4 # Large enough for the streak
            
            if cache_key in self.kernel_cache:
                phosphene = self.kernel_cache[cache_key]
            else:
                # Sigma varies with brightness (blooming)
                sigma = (rf_size * out_scale * 0.5) * (0.5 + 0.5 * perceived_lum)
                phosphene = self._elongated_gaussian(kernel_size, perceived_lum, sigma, angle, self.axon_lambda)
                self.kernel_cache[cache_key] = phosphene
            
            # Image composition (blending)
            # Compute output center
            out_r, out_c = int(pt[0] * out_h), int(pt[1] * out_w)
            
            k_h, k_w = phosphene.shape
            r_start = out_r - k_h // 2
            c_start = out_c - k_w // 2
            
            # Manual border handling (clipping)
            r_min, r_max = max(0, r_start), min(out_h, r_start + k_h)
            c_min, c_max = max(0, c_start), min(out_w, c_start + k_w)
            
            pr_min = r_min - r_start
            pr_max = pr_min + (r_max - r_min)
            pc_min = c_min - c_start
            pc_max = pc_min + (c_max - c_min)
            
            out_image[r_min:r_max, c_min:c_max] = np.maximum(
                out_image[r_min:r_max, c_min:c_max],
                phosphene[pr_min:pr_max, pc_min:pc_max]
            )
            
        return out_image