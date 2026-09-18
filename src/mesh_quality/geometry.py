"""Surface sampling and local 3D architecture used in the research baseline."""

import hashlib

import numpy as np
import torch
from torch import nn

NUM_POINTS = 8192
FACE_CHUNK_SIZE = 100_000


def item_sampling_seed(item_id, seed):
    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
    )


def read_triangles(vertices, face_block):
    if face_block.min() < 0 or face_block.max() >= len(vertices):
        raise ValueError("Грань ссылается на несуществующую вершину.")

    triangles = np.asarray(
        vertices[face_block],
        dtype=np.float64,
    )

    if not np.isfinite(triangles).all():
        raise ValueError("Координаты используемых вершин содержат NaN/inf.")

    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]

    cross_products = np.cross(edge_1, edge_2)

    areas = 0.5 * np.linalg.norm(
        cross_products,
        axis=1,
    )

    if not np.isfinite(areas).all():
        raise ValueError("Не удалось вычислить конечные площади граней.")

    return triangles, areas


def sample_mesh_surface(
    npz_path,
    item_id,
    num_points=8192,
    seed=42,
    face_chunk_size=100_000,
):
    if num_points <= 0 or face_chunk_size <= 0:
        raise ValueError("Число точек и размер блока должны быть положительными.")

    with np.load(npz_path, allow_pickle=False) as data:
        vertices = data["vertices"]
        faces = data["faces"]

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Ожидаются непустые vertices формы [N, 3].")

    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Ожидаются непустые треугольные faces формы [M, 3].")

    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Индексы граней должны быть целыми числами.")

    chunk_starts = list(range(0, len(faces), face_chunk_size))

    chunk_areas = []

    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)

    zero_area_faces = 0

    # Первый проход: площади и границы геометрии.
    for start in chunk_starts:
        triangles, areas = read_triangles(
            vertices,
            faces[start : start + face_chunk_size],
        )

        lower = np.minimum(
            lower,
            triangles.min(axis=(0, 1)),
        )
        upper = np.maximum(
            upper,
            triangles.max(axis=(0, 1)),
        )

        chunk_areas.append(areas.sum())
        zero_area_faces += int((areas == 0).sum())

    chunk_areas = np.asarray(
        chunk_areas,
        dtype=np.float64,
    )

    total_area = chunk_areas.sum()

    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Нет поверхности с конечной положительной площадью.")

    # Нормализация с сохранением пропорций.
    # Учитываем вершины, на которые ссылаются грани.
    center = lower / 2 + upper / 2
    scale = float((upper - lower).max())

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Некорректный размер меша.")

    rng = np.random.default_rng(item_sampling_seed(item_id, seed))

    chosen_chunks = rng.choice(
        len(chunk_starts),
        size=num_points,
        p=chunk_areas / total_area,
    )

    face_ids = np.empty(
        num_points,
        dtype=np.int64,
    )

    # Второй проход: выбираем грани внутри нужных блоков.
    for chunk_index in np.unique(chosen_chunks):
        positions = np.flatnonzero(chosen_chunks == chunk_index)

        start = chunk_starts[chunk_index]

        _, areas = read_triangles(
            vertices,
            faces[start : start + face_chunk_size],
        )

        local_ids = rng.choice(
            len(areas),
            size=len(positions),
            p=areas / areas.sum(),
        )

        face_ids[positions] = start + local_ids

    selected_triangles = (
        np.asarray(
            vertices[faces[face_ids]],
            dtype=np.float64,
        )
        - center
    ) / scale

    # Равномерные точки внутри треугольников.
    random_values = rng.random((num_points, 2))

    root_u = np.sqrt(random_values[:, 0])
    v = random_values[:, 1]

    points = (
        (1 - root_u)[:, None] * selected_triangles[:, 0]
        + (root_u * (1 - v))[:, None] * selected_triangles[:, 1]
        + (root_u * v)[:, None] * selected_triangles[:, 2]
    )

    # Нормаль исходной грани — без сглаживания.
    normals = np.cross(
        selected_triangles[:, 1] - selected_triangles[:, 0],
        selected_triangles[:, 2] - selected_triangles[:, 0],
    )

    normals /= np.linalg.norm(
        normals,
        axis=1,
        keepdims=True,
    )

    points = points.astype(np.float32)
    normals = normals.astype(np.float32)

    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("Получены некорректные точки или нормали.")

    return {
        "points": points,
        "normals": normals,
        "face_ids": face_ids,
        "center": center,
        "scale": scale,
        "num_vertices": len(vertices),
        "num_faces": len(faces),
        "zero_area_faces": zero_area_faces,
        "surface_area": float(total_area),
        "seed": seed,
        "face_chunk_size": face_chunk_size,
        "item_id": str(item_id),
    }


@torch.no_grad()
def chunked_knn(
    xyz,
    queries,
    k,
    chunk_size=64,
):
    if not 1 <= k <= xyz.shape[1] or chunk_size < 1:
        raise ValueError("Некорректные k или chunk_size.")

    source_norm = xyz.square().sum(dim=-1).unsqueeze(1)

    source_transposed = xyz.transpose(1, 2)

    index_blocks = []

    for start in range(0, queries.shape[1], chunk_size):
        query_block = queries[
            :,
            start : start + chunk_size,
        ]

        # Квадраты евклидовых расстояний:
        # ||q - x||² = ||q||² + ||x||² - 2(q · x).
        distances = (
            query_block.square().sum(
                dim=-1,
                keepdim=True,
            )
            + source_norm
            - 2
            * torch.bmm(
                query_block,
                source_transposed,
            )
        )

        distances.clamp_min_(0)

        indices = distances.topk(
            k,
            dim=-1,
            largest=False,
            sorted=False,
        ).indices

        index_blocks.append(indices)

    return torch.cat(index_blocks, dim=1)


class LocalGroupingBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        num_centers,
        neighbors,
        hidden_dim,
        output_dim,
        chunk_size=64,
    ):
        super().__init__()

        self.num_centers = num_centers
        self.neighbors = neighbors
        self.chunk_size = chunk_size

        self.local_mlp = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, xyz, features):
        if xyz.shape[1] < self.num_centers:
            raise ValueError("Недостаточно точек для выбранного числа центров.")

        centers = xyz[:, : self.num_centers]

        indices = chunked_knn(
            xyz=xyz,
            queries=centers,
            k=self.neighbors,
            chunk_size=self.chunk_size,
        )

        batch_indices = torch.arange(
            xyz.shape[0],
            device=xyz.device,
        )[:, None, None]

        neighbor_xyz = xyz[
            batch_indices,
            indices,
        ]

        neighbor_features = features[
            batch_indices,
            indices,
        ]

        relative_xyz = neighbor_xyz - centers.unsqueeze(2)

        local_inputs = torch.cat(
            [
                relative_xyz,
                neighbor_features,
            ],
            dim=-1,
        )

        local_features = self.local_mlp(local_inputs).amax(dim=2)

        return centers, local_features


class LocalGeometryClassifier(nn.Module):
    def __init__(
        self,
        num_defects,
        centers=(512, 128),
        neighbors=(32, 32),
        chunk_size=64,
        dropout=0.2,
    ):
        super().__init__()

        self.stage1 = LocalGroupingBlock(
            input_dim=3,
            num_centers=centers[0],
            neighbors=neighbors[0],
            hidden_dim=64,
            output_dim=96,
            chunk_size=chunk_size,
        )

        self.stage2 = LocalGroupingBlock(
            input_dim=96,
            num_centers=centers[1],
            neighbors=neighbors[1],
            hidden_dim=128,
            output_dim=192,
            chunk_size=chunk_size,
        )

        self.global_projection = nn.Sequential(
            nn.Linear(192 + 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_defects),
        )

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[-1] != 6:
            raise ValueError("Ожидается [batch, points, 6].")

        xyz = inputs[..., :3]
        normals = inputs[..., 3:]

        xyz1, features1 = self.stage1(xyz, normals)

        xyz2, features2 = self.stage2(xyz1, features1)

        object_tokens = self.global_projection(
            torch.cat(
                [xyz2, features2],
                dim=-1,
            )
        )

        pooled = torch.cat(
            [
                object_tokens.mean(dim=1),
                object_tokens.amax(dim=1),
            ],
            dim=-1,
        )

        return self.classifier(pooled)
