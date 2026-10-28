import numpy as np
import cv2
import math

class PhospheneRenderer:
    def __init__(self, electrode_grid=(32, 32), lattice_structure=(1, 0, 1), 
                 dropout_rate=0.1, intensity_levels=8, 
                 axon_lambda=2.0, optic_disc_loc=(0.8, 0.5)):
        """
        Renderer SPV Biologicamente Plausibile (Axon Map Model).
        
        Ref: Kasowski & Beyeler (2022); Davis (2020).
        
        Args:
            electrode_grid: (cols, rows) number of electrodes
            lattice_structure: parametri spaziatura
            dropout_rate: Fraction of dead electrodes (simulated failures)
            intensity_levels: Discrete grayscale levels
            axon_lambda: Elongation factor of the 'comet' (axonal streaks)
            optic_disc_loc: Normalized (x, y) coordinates of the optic disc (blind spot)
        """
        self.n_elec_x = electrode_grid[0]
        self.n_elec_y = electrode_grid[1]
        self.lattice = lattice_structure
        self.levels = intensity_levels
        self.axon_lambda = axon_lambda
        self.optic_disc = optic_disc_loc
        
        # 1. Generate electrode positions
        self.centers = self._generate_hex_grid()
        
        # 2. Dropout mask (broken electrodes)
        num_electrodes = len(self.centers)
        self.active_mask = np.random.rand(num_electrodes) > dropout_rate
        
        # 3. PRE-COMPUTE AXONAL KERNELS (the core of the optimization)
        # Instead of computing the rotation every frame, we generate the base image
        # of every single electrode, distorted towards the optic disc.
        self.electrode_kernels = []
        self._precompute_axon_kernels()

    def _generate_hex_grid(self):
        # (Same hexagonal grid generation as the previous version)
        v1 = self.lattice[0] / self.n_elec_x
        v2 = self.lattice[1] / self.n_elec_x
        v3 = self.lattice[2] / self.n_elec_y
        
        x_array = np.linspace(v1/2, 1 - v1/2, self.n_elec_x)
        x_array_d = np.linspace(v1/2 + v2, 1 - v1/2 + v2, self.n_elec_x)
        y_array = np.linspace(v3/2, 1 - v3/2, self.n_elec_y)
        
        centers = []
        for i in range(self.n_elec_y):
            y_coord = y_array[i]
            current_x_array = x_array_d if (i % 2 == 0 and v2 > 0) else x_array
            for x_coord in current_x_array:
                centers.append([y_coord, x_coord]) # [row, col]
        return np.array(centers)

    def _oriented_gaussian(self, size, sigma_x, sigma_y, angle):
        """Generate a rotated Gaussian (ellipse) to simulate the nerve fiber bundle."""
        size = int(size)
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Rotazione coordinate
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        X_rot = X * cos_a + Y * sin_a
        Y_rot = -X * sin_a + Y * cos_a
        
        # Formula Gaussiana Ellittica
        return np.exp(-( (X_rot**2)/(2*sigma_x**2) + (Y_rot**2)/(2*sigma_y**2) ))

    def _precompute_axon_kernels(self):
        """
        Pre-compute the image of every electrode.
        According to Kasowski (2022), axons curve towards the optic disc.
        We approximate this curve with ellipses oriented towards self.optic_disc.
        """
        # Fixed (scaled) kernel size per electrode
        kernel_size = 32 
        
        for pt in self.centers:
            # pt is [y, x] normalized (0-1)
            ey, ex = pt
            
            # Compute the vector towards the optic disc
            dy = ey - self.optic_disc[1]
            dx = ex - self.optic_disc[0]
            
            # Angle towards the optic disc
            angle = np.arctan2(dy, dx)
            
            # Parametri forma (Model Beyeler/Kasowski)
            # sigma_x = larghezza punto (stretto)
            # sigma_y = streak length (long, depends on lambda)
            # The higher lambda, the longer the streak
            base_sigma = 0.15
            sigma_short = base_sigma
            sigma_long = base_sigma * self.axon_lambda 
            
            kernel = self._oriented_gaussian(kernel_size, sigma_short, sigma_long, angle)
            self.electrode_kernels.append(kernel)

    def render(self, image):
        """Render a frame with axon map and temporal noise."""
        # 1. Preprocessing: Edge Enhancement (Kasowski 2022 raccomanda Sobel) 
        # Helps define the structure before degrading the image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Scale the input
        if gray.dtype == np.uint8:
            gray = gray.astype(float) / 255.0
            
        # Apply light edge enhancement
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        gray = np.clip(gray + 0.3 * np.abs(edges), 0, 1)

        h_img, w_img = gray.shape
        
        # High-resolution output (to show the elliptical shapes)
        scale_out = 8
        out_h, out_w = h_img * scale_out, w_img * scale_out
        out_image = np.zeros((out_h, out_w), dtype=float)
        
        # Dimensioni Receptive Field Input
        rf_h = int(h_img / self.n_elec_y / 1.5)
        rf_w = int(w_img / self.n_elec_x / 1.5)
        
        # Loop rendering
        for i, pt in enumerate(self.centers):
            if not self.active_mask[i]: continue
            
            # Coordinate Input
            r_in = int(pt[0] * h_img)
            c_in = int(pt[1] * w_img)
            
            # Extract local mean brightness
            r1, r2 = max(0, r_in-rf_h), min(h_img, r_in+rf_h)
            c1, c2 = max(0, c_in-rf_w), min(w_img, c_in+rf_w)
            patch = gray[r1:r2, c1:c2]
            
            if patch.size == 0: val = 0
            else: val = np.mean(patch)
            
            # Quantization (discrete levels)
            if self.levels > 0:
                val = np.floor(val * self.levels) / self.levels
            
            if val < 0.05: continue # Minimum activation threshold
            
            # 2. Add temporal noise ("Sparkles" / "Glitter")
            # Davis (2020) reports that patients perceive "sparkles" and instability.
            # Add a random ±15% flicker
            noise = np.random.uniform(0.85, 1.15)
            val = np.clip(val * noise, 0, 1)
            
            # 3. Draw the pre-computed kernel (axon streak)
            kernel = self.electrode_kernels[i]
            k_h, k_w = kernel.shape
            
            # Coordinate Output
            center_out_r = int(pt[0] * out_h)
            center_out_c = int(pt[1] * out_w)
            
            # Compute the bounding box where the kernel is added
            r_start = center_out_r - k_h // 2
            c_start = center_out_c - k_w // 2
            r_end = r_start + k_h
            c_end = c_start + k_w
            
            # Border handling (clipping)
            k_r_s, k_r_e = 0, k_h
            k_c_s, k_c_e = 0, k_w
            
            if r_start < 0: k_r_s = -r_start; r_start = 0
            if c_start < 0: k_c_s = -c_start; c_start = 0
            if r_end > out_h: k_r_e -= (r_end - out_h); r_end = out_h
            if c_end > out_w: k_c_e -= (c_end - out_w); c_end = out_w
            
            if r_end > r_start and c_end > c_start:
                # Additive blending (sum of neighbouring electrode contributions)
                out_image[r_start:r_end, c_start:c_end] += (kernel[k_r_s:k_r_e, k_c_s:k_c_e] * val)

        return np.clip(out_image, 0, 1)