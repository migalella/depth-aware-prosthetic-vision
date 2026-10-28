import numpy as np
import cv2

class PhospheneRenderer:
    def __init__(self, electrode_grid=(32, 32), lattice_structure=(1, 0.5, 1), 
                 dropout_rate=0.1, intensity_levels=8, 
                 axon_lambda=2.0, temporal_decay=0.1):
        
        # The electrode_grid parameter indicates the resolution of the array.
        # The dropout rate simulates that ~10% of electrodes might be broken or not contacting the retina.
        # The axon_lambda parameter controls how elongated the phosphenes are (simulating axonal activation).
        # The temporal_decay parameter simulates the retina desensitizing over time (fading).        
        
        self.n_elec_x = electrode_grid[0]
        self.n_elec_y = electrode_grid[1]
        self.levels = intensity_levels
        self.axon_lambda = axon_lambda
        self.temporal_decay = temporal_decay
        
        # 1. Grid generation (Jittered Hexagonal), since real implants are not perfectly regular.
        base_centers = self._generate_hex_grid(lattice_structure)
        noise = (np.random.rand(*base_centers.shape) - 0.5) * 0.02 
        self.centers = base_centers + noise
        
        # 2. Dropout mask: boolean array to decide which electrodes are active.
        self.active_mask = np.random.rand(len(self.centers)) > dropout_rate
        
        # 3. Axon map: precompute orientations for each electrode.
        self.orientations = self._calculate_axon_map()
        
        # 4. Temporal cache: store previous activation levels for fading effect.
        self.prev_activation = np.zeros(len(self.centers))
        
        # 5. Precompute LUT: optimization to avoid recalculating phosphenes every frame. It precomputes phosphenes for all intensity levels.
        self.lut = self._precompute_lut()

    def _generate_hex_grid(self, lattice):
        # It calculates the coordinates for every electrode, implementing a jittered hexagonal grid.
        # Every odd row is staggered by half the electrode spacing, as in the standard hexagonal lattice structures used in prosthetics.
        ratio_x, stagger, ratio_y = lattice
        step_x = 1.0 / self.n_elec_x
        step_y = 1.0 / self.n_elec_y
        centers = []
        for r in range(self.n_elec_y):
            y = r * step_y + (step_y / 2)
            row_stagger = (step_x * stagger) if (r % 2 == 1) else 0
            for c in range(self.n_elec_x):
                x = c * step_x + (step_x / 2) + row_stagger
                if 0 <= x <= 1 and 0 <= y <= 1:
                    centers.append([y, x])
        return np.array(centers)

    def _calculate_axon_map(self):

        # It calculates the angle of the nerve fiber bundles at each electrode position, relative to a fixed point (the optic disc).
        # In epiretinal implants, stimulation often activates axons passing under the electrode. These axons run toward the optic nerve. 
        # Consequently, patients often see elongated streaks or "matchsticks" aligned with this path, rather than perfect dots.

        angles = []
        optic_disc = np.array([0.5, -0.1]) 
        for pt in self.centers:
            dy = pt[0] - optic_disc[0]
            dx = pt[1] - optic_disc[1]
            angles.append(np.arctan2(dy, dx))
        return np.array(angles)

    def _elongated_gaussian(self, size, luminance, sigma_base, angle, elongation):

        if luminance <= 0.001: return None
        
        # Blooming: it modifies the width of the Gaussian (current_sigma) based on luminance. 
        # Brighter phosphenes appear physically larger, a psychophysical phenomenon documented in patient trials
        current_sigma = sigma_base * (0.8 + 0.7 * luminance)
        
        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)
        
        c, s = np.cos(angle), np.sin(angle)
        X_rot = c * X - s * Y
        Y_rot = s * X + c * Y
        
        # Gaussian formula: it creates a pixel matrix representing a single point of light. 
        # It uses a 2D Gaussian function but with different widths for X and Y (elongation) and rotates it. 
        # This creates the biologically accurate elliptical/streak shape.
        sigma_x = current_sigma
        sigma_y = current_sigma * elongation
        
        gauss = luminance * np.exp(-((X_rot**2)/(2*sigma_x**2) + (Y_rot**2)/(2*sigma_y**2)))
        
        # Background noise cleanup
        gauss[gauss < 0.005] = 0
        return gauss

    def _precompute_lut(self):
        # It precomputes the phosphene shapes for each electrode at every intensity level, storing them in a lookup table (LUT).
        lut = []
        rf_size = 32 # Base phosphene size
        k_size = int(rf_size * self.axon_lambda * 1.5)
        if k_size % 2 == 0: k_size += 1
        
        for idx, pt in enumerate(self.centers): # For each electrode
            elec_lut = []
            if not self.active_mask[idx]:
                lut.append(None)
                continue
            
            angle = self.orientations[idx]
            
            for level in range(self.levels + 1): # For each intensity level
                lum = level / self.levels
                if lum == 0:
                    elec_lut.append(None)
                else:
                    kernel = self._elongated_gaussian(k_size, lum, rf_size/4, angle, self.axon_lambda)
                    elec_lut.append(kernel)
            lut.append(elec_lut)
        return lut

    def render(self, image):
        
        # 1. Resize input image to electrode grid size.
        # cv2.INTER_AREA computes the relationship between pixel areas. 
        # Each "pixel" of the new 32x32 image becomes the average brightness of the corresponding area in the original image. This corresponds to the receptive field (aperture) of the phosphene.
        stimulus_lowres = cv2.resize(image, (self.n_elec_x, self.n_elec_y), interpolation=cv2.INTER_AREA)
        
        # 2. Flatten and normalize the stimulus to [0,1].
        # The conversion in float is essential for subsequent mathematical operations, such as normalization (division by 255.0) and temporal decay calculation. 
        # If they remained integers, operations like pixel / 255 would result in 0 for almost all values.
        stimulus_flat = stimulus_lowres.flatten().astype(float)
        
        # Robust normalization: if input is 0-255 convert, if 0-1 keep as is.
        if stimulus_flat.max() > 1.1:
            activations = stimulus_flat / 255.0
        else:
            activations = stimulus_flat

        # Contrast enhancement: it slightly boosts mid-level brightness values to improve visibility of phosphenes.
        activations = np.clip(activations * 1.2, 0, 1)

        # 3. Temporal decay application (desensitization).
        # It calculates the output by subtracting a fraction of the previous frame's activation. 
        # This simulates neural adaptation. If you stare at a static light, neurons stop firing, and the phosphene fades. 
        # This forces the system to react to change (movement) rather than static light, increasing biological validity regarding flicker fusion and persistence.
        effective_activation = activations
        if self.temporal_decay > 0:
            change = activations - (self.prev_activation * self.temporal_decay)
            effective_activation = np.clip(change, 0, 1)
        
        self.prev_activation = activations
        
        # 4. Quantization into discrete levels for LUT indexing.
        indices = (effective_activation * self.levels).astype(int)
        indices = np.clip(indices, 0, self.levels)
        
        # 5. Final Rendering
        out_size = 256
        output_canvas = np.zeros((out_size, out_size), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            if idx == 0 or self.lut[i] is None: continue
            
            kernel = self.lut[i][idx]
            if kernel is None: continue
            
            cy = int(self.centers[i][0] * out_size)
            cx = int(self.centers[i][1] * out_size)
            
            kh, kw = kernel.shape
            r_start = cy - kh // 2
            c_start = cx - kw // 2
            
            r_min, r_max = max(0, r_start), min(out_size, r_start + kh)
            c_min, c_max = max(0, c_start), min(out_size, c_start + kw)
            
            kr_min = r_min - r_start
            kc_min = c_min - c_start
            
            if r_max > r_min and c_max > c_min:
                target_slice = output_canvas[r_min:r_max, c_min:c_max]
                kernel_slice = kernel[kr_min:kr_min+(r_max-r_min), kc_min:kc_min+(c_max-c_min)]
                # If two phosphenes overlap, it takes the maximum brightness value rather than adding them (which would cause unrealistic white saturation). 
                # This simulates fusion via saturated addition, ensuring adjacent phosphenes merge organically.
                np.maximum(target_slice, kernel_slice, out=target_slice)

        return output_canvas