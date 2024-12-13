from sentence_transformers import SentenceTransformer, models
import torch

class RoLaserSentenceEncoder(SentenceTransformer):
    def __init__(self, model_name_or_path, laser_embed_dim=1024, *model_args, **kwargs):
        transformer = models.Transformer(model_name_or_path)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        dense = models.Dense(
            in_features=transformer.get_word_embedding_dimension(), 
            out_features=laser_embed_dim, 
            bias=True, 
            activation_function=torch.nn.GELU()
        )
        super().__init__(modules=[transformer, pooling, dense], *model_args, **kwargs)
    