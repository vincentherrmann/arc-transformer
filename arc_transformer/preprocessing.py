import torch
import torch.nn.functional as F
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, features_dim, highest_freq=0.25):
        super().__init__()
        frequencies = torch.exp(torch.linspace(0., -10., features_dim // 4)) * highest_freq
        self.register_buffer('frequencies', frequencies)
        self.s = math.pi * 2

    def forward(self, x):
        # x:  th x tw x h x w x batch
        w_cos_encoding = torch.cos(self.s * x[:, None] * self.frequencies[None, :])
        w_sin_encoding = torch.sin(self.s * x[:, None] * self.frequencies[None, :])
        encoding = torch.cat([cos_encoding, sin_encoding], dim=1)
        return encoding


class ArcPreprocessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_dict):
        max_l = 0
        for key, value in data_dict.items():
            for k, v, in value.items():
                w, h = v.shape
                max_l = max(max_l, w*h)

                features = F.one_hot(v, num_classes=10)
                pos_w = torch.arange(w)[:, None].repeat(1, h)
                pos_h = torch.arange(h)[None, :].repeat(w, 1)


class PriorCalculation(torch.nn.Module):
    def __init__(self, max_width=30, max_height=30, num_colors=10, max_num_objects=8, highes_freq=0.25,
                 num_positional_features=64):
        super().__init__()
        self.th = 2*max_height - 1
        self.tw = 2*max_width - 1
        self.ch = max_height - 1
        self.cw = max_width - 1
        self.patterns = self.create_attention_patterns(size=(self.th, self.tw))
        #self.patterns = self.patterns.view(-1, self.patterns.shape[2])
        self.num_colors = num_colors
        self.max_num_objects = max_num_objects
        self.num_priors = self.patterns.shape[1] + 2 + self.max_num_objects * 8
        self.highest_freq = highes_freq
        self.num_positional_features = num_positional_features
        self.positional_encoding = self.create_positional_encoding()
        pass

    def get_relative_priors(self, x):
        if len(x.shape) == 4:
            x = x[0]
        h, w, _ = x.shape

        relative_priors = self.positional_encoding[self.ch - h + 1:self.ch + h, self.cw - w + 1:self.cw + w]
        return relative_priors

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[0]
        h, w = x.shape

        #absolute_priors = self.positional_encoding[self.ch:self.ch + h, self.cw:self.cw + w]

        x = F.one_hot(x, num_classes=10).float()
        #x = torch.cat([x, absolute_priors], dim=2)

        return x

        th = 2*h - 1
        tw = 2*w - 1

        # object prior
        objects = self.extract_objects(x)
        objects.sort(key=lambda o: o[0].sum(), reverse=True)
        objects = objects[:self.max_num_objects]
        if len(objects) == 0:
            complete_objects_prior = torch.zeros_like(color_prior)
            extracted_objects_prior = torch.zeros(h, w, h, w, self.max_num_objects * 8).to(complete_objects_prior.device)
        else:
            complete_objects = torch.stack([o[0] for o in objects], dim=2).view(h, w, -1)  # s x o
            complete_objects_prior = torch.einsum('so,to->st', complete_objects, complete_objects).view(h, w, h, w, 1).float()

            extracted_objects = []
            max_size = 0
            for _, obj in objects:
                max_size = max(max_size, max(obj.shape))
                extracted_objects.append(obj)
                extracted_objects.append(torch.flip(obj, dims=[0]))
                extracted_objects.append(torch.flip(obj, dims=[1]))
                extracted_objects.append(torch.flip(obj, dims=[0, 1]))
                extracted_objects.append(obj.t())
                extracted_objects.append(torch.flip(obj.t(), dims=[0]))
                extracted_objects.append(torch.flip(obj.t(), dims=[1]))
                extracted_objects.append(torch.flip(obj.t(), dims=[0, 1]))

            object_canvas = torch.zeros(max_size, max_size).long()
            for i, obj in enumerate(extracted_objects):
                c = object_canvas.clone()
                c[:obj.shape[0], :obj.shape[1]] = obj
                extracted_objects[i] = c
            extracted_objects = torch.stack(extracted_objects, dim=0)
            c = torch.zeros(extracted_objects.shape[0], h, w).long()
            c[:, :max_size, :max_size] = extracted_objects[:, :h, :w]
            skew_c = torch.zeros(c.shape[0], h*w+1).long()
            skew_c[:, :h*w] = c.view(-1, h*w)
            skew_c = skew_c[:, None, :].repeat(1, h*w, 1).view(-1, h*w+1, h*w)
            extracted_objects_prior = skew_c[:, :h*w, :].view(-1, h, w, h, w).permute(1, 2, 3, 4, 0).float()
            if extracted_objects_prior.shape[4] < self.max_num_objects * 8:
                extracted_objects_prior = F.pad(extracted_objects_prior,
                                                pad=(0, self.max_num_objects * 8 - extracted_objects_prior.shape[4]))

        all_priors = torch.cat([complete_objects_prior, extracted_objects_prior], dim=4)
        return all_priors.view(h*w, h*w, -1)

    def create_positional_encoding(self):
        freqs = torch.exp(torch.linspace(0., -3., self.num_positional_features // 4)) * self.highest_freq
        s = math.pi * 2

        pos = torch.arange(self.th) - self.ch
        cos_encoding = torch.cos(s * pos[:, None] * freqs[None, :])
        sin_encoding = torch.sin(s * pos[:, None] * freqs[None, :])
        h_encoding = torch.cat([cos_encoding, sin_encoding], dim=1)[:, None, :].repeat(1, self.tw, 1)

        pos = torch.arange(self.tw) - self.cw
        cos_encoding = torch.cos(s * pos[:, None] * freqs[None, :])
        sin_encoding = torch.sin(s * pos[:, None] * freqs[None, :])
        w_encoding = torch.cat([cos_encoding, sin_encoding], dim=1)[None, :, :].repeat(self.th, 1, 1)
        combined_encodings = torch.cat([h_encoding, w_encoding], dim=2)
        return combined_encodings  # th x tw x e

    @staticmethod
    def create_attention_patterns(size=(9, 9)):
        c_w = (size[0] - 1) // 2
        c_h = (size[1] - 1) // 2
        indices = torch.arange(size[0])
        rev_indices = torch.flip(indices, dims=[0])
        dist = indices - c_w
        position = torch.stack([dist[:, None].repeat(1, size[1]),
                                dist[None, :].repeat(size[1], 1)], dim=0)

        patterns = []
        horizontal_line = torch.zeros(size)
        horizontal_line[:, c_h] = 1.
        patterns.append(horizontal_line)

        vertical_line = torch.zeros(size)
        vertical_line[c_w, :] = 1.
        patterns.append(vertical_line)

        diagonal_line_1 = torch.zeros(size)
        diagonal_line_1[indices, indices] = 1.
        patterns.append(diagonal_line_1)

        diagonal_line_2 = torch.zeros(size)
        diagonal_line_2[indices, rev_indices] = 1.
        patterns.append(diagonal_line_2)

        checkerboard_1 = torch.zeros(size[0] * size[1])
        checkerboard_1[::2] = 1.
        checkerboard_1 = checkerboard_1.view(size)
        patterns.append(checkerboard_1)

        checkerboard_1_r = 1 - checkerboard_1
        patterns.append(checkerboard_1_r)

        distance = torch.abs(position[0]) + torch.abs(position[1])
        patterns.append(distance.float())

        concentric_squares = torch.max(torch.abs(position), dim=0)[0]
        patterns.append(concentric_squares.float())

        crosses = torch.min(torch.abs(position), dim=0)[0]
        patterns.append(crosses.float())

        for i in range(2, 5):
            stripes = ((dist % i) == 0).float()[:, None].repeat(1, size[1])
            patterns.append(stripes)
            patterns.append(stripes.t())

        position_vertical = dist[:, None].repeat(1, size[1]).float()
        patterns.append(position_vertical)
        patterns.append(c_w - position_vertical)
        patterns.append(position_vertical.t())
        patterns.append(c_w - position_vertical.t())

        quadrant = ((position[0] >= 0.) * (position[1] >= 0.)).float()
        patterns.append(quadrant)
        patterns.append(torch.flip(quadrant, dims=[0]))
        patterns.append(torch.flip(quadrant, dims=[1]))
        patterns.append(torch.flip(quadrant, dims=[0, 1]))

        patterns = torch.stack(patterns, dim=2)
        return patterns  #.view(size[0], size[1], -1)

    def find_object(self, grid, current_object, x, y):
        if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
            return
        v = grid[x, y]
        if v == 0:
            return
        grid[x, y] = 0.
        current_object[x, y] = 1.

        self.find_object(grid, current_object, x - 1, y)
        self.find_object(grid, current_object, x + 1, y)
        self.find_object(grid, current_object, x, y - 1)
        self.find_object(grid, current_object, x, y + 1)

    def extract_objects(self, grid):
        objects = []
        empty_object = torch.zeros_like(grid)
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                v = grid[x, y]
                if v == 0:
                    continue
                current_object = empty_object.clone()
                self.find_object(grid, current_object, x, y)

                if torch.sum(current_object).item() > 1.:
                    objects.append(current_object)

        extracted_objects = []
        for obj in objects:
            x_range = torch.sum(obj, dim=1) > 0
            y_range = torch.sum(obj, dim=0) > 0
            obj = obj[x_range, :]
            obj = obj[:, y_range]
            extracted_objects.append(obj)
        return [(objects[i], extracted_objects[i]) for i in range(len(objects))]
