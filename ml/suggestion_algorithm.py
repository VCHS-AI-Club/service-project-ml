from typing import Dict, Text
import tensorflow as tf

import numpy as np
import tensorflow_recommenders as tfrs

# sample data, randomly generated
ratings = np.random.randint(10, size=(100))
users = np.random.randint(100, size=(100))

opp = np.array([])
for i in range(100):
  opp = np.append(opp, "some text{}".format(i))

ratings_dataset = {i:j for (i, j) in zip(opp, users)}

user_id_vocab = tf.keras.layers.StringLookup(mask_token=None)
user_id_vocab.adapt(ratings_dataset.keys())

opp_titles_vocab = tf.keras.layers.StringLookup(mask_token=None)
opp_titles_vocab.adapt(opp)

class OpportunityModel(tfrs.Model):

  def __init__(
      self,
      user_model: tf.keras.Model,
      opp_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    self.user_model = user_model
    self.opp_model = opp_model

    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    user_embeddings = self.user_model(features["user_id"])
    opp_embeddings = self.movie_model(features["opp"])

    return self.task(user_embeddings, opp_embeddings)

embedding_size = 32 
batch_size = 10 

user_model = tf.keras.Sequential([
    user_id_vocab,
    tf.keras.layers.Embedding(user_id_vocab.vocabulary_size(), embedding_size)
])
opp_model = tf.keras.Sequential([
    opp_titles_vocab,
    tf.keras.layers.Embedding(opp_titles_vocab.vocabulary_size(), embedding_size)
])

task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    opp.batch(batch_size).map(opp_model)
  )
)

model = OpportunityModel(user_model, opp_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

model.fit(ratings.batch(10), epochs=5)

index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    opp.batch(100).map(lambda title: (title, model.opp_model(title))))

query :str = "42"    

_, opp_list = index(np.array([query]))
print(f"Top 5 recommendations for user {query}: {opp_list[0, :5]}")
