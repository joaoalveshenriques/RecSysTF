#!/usr/bin/env python
# coding: utf-8

# In[102]:


masterdf.info()


# In[104]:


purchases_dict = masterdf.groupby([
                                            'user_id',
                                            'user_gender',
                                            'user_age',
                                            'user_lat',
                                            'user_long',
                                            'timestamp',
                                            'product_id',
                                            'price',
                                            'product_colour',
                                            'product_tear',
                                            'product_tonality',
                                            'product_gender',
                                            'product_age',
                                            'product_category',
                                            'product_fit',
                                            'product_rise',
                                            'product_neckline',
                                            'product_sleeve',
                                            'product_denim',
                                            'product_stretch',
                                            'product_wash',
                                            'user_rating'
    
])['quantity'].sum().reset_index()

purchases_dict = {name: np.array(value) for name, value in purchases_dict.items()}
purchases = tf.data.Dataset.from_tensor_slices(purchases_dict)

products_dict = masterdf[['product_id']].drop_duplicates().dropna()
products_dict = {name: np.array(value) for name, value in products_dict.items()}
products = tf.data.Dataset.from_tensor_slices(products_dict)

purchases = purchases.map(lambda x: {
                                            'user_id' : x['user_id'], 
                                            'user_gender' : x['user_gender'],
                                            'user_age' : x['user_age'],
                                            'user_lat' : x['user_lat'],
                                            'user_long' : x['user_long'],   
                                            'product_id' : x['product_id'],
                                            'quantity': x['quantity'],
                                            'price' : x['price'],
                                            'timestamp': x['timestamp'],
                                            'product_colour': x['product_colour'],
                                            'product_tear' : x['product_tear'], 
                                            'product_tonality' : x['product_tonality'],
                                            'product_gender': x['product_gender'],
                                            'product_age' : x['product_age'], 
                                            'product_category' : x['product_category'],
                                            'product_fit': x['product_fit'],
                                            'product_rise' : x['product_rise'], 
                                            'product_neckline' : x['product_neckline'],
                                            'product_sleeve': x['product_sleeve'],
                                            'product_denim' : x['product_denim'], 
                                            'product_stretch' : x['product_stretch'],
                                            'product_wash': x['product_wash'],
                                            'user_rating': x['user_rating']
})

products = products.map(lambda x: x['product_id'])


# In[105]:


product_ids = products.batch(1_000)
user_ids = purchases.batch(1_000_000).map(lambda x: x["user_id"])

unique_product_ids = np.unique(np.concatenate(list(product_ids)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))


# In[106]:


unique_user_gender = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["user_gender"]))))
unique_product_colour = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_colour"]))))
unique_product_tear = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_tear"]))))
unique_product_tonality = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_tonality"]))))
unique_product_gender = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_gender"]))))
unique_product_age = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_age"]))))
unique_product_category = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_category"]))))
unique_product_fit = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_fit"]))))
unique_product_rise = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_rise"]))))
unique_product_neckline = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_neckline"]))))
unique_product_sleeve = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_sleeve"]))))
unique_product_denim = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_denim"]))))
unique_product_stretch = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_stretch"]))))
unique_product_wash = np.unique(np.concatenate(list(purchases.batch(1_000_000).map(lambda x: x["product_wash"]))))


# In[107]:


timestamps = list(purchases.map(lambda x: x["timestamp"]).batch(100))

timestamp_buckets = np.linspace(
    np.concatenate(list(purchases.map(lambda x: x["timestamp"]).batch(100))).min(), 
    np.concatenate(list(purchases.map(lambda x: x["timestamp"]).batch(100))).max(), num=1000,
)

user_age = list(purchases.map(lambda x: x["user_age"]).batch(100))

user_age = list(purchases.map(lambda x: x["user_age"]).batch(100))

user_age_buckets = np.linspace(
    np.concatenate(user_age).min(), 
    np.concatenate(user_age).max(), num=1000,
)

latitude = list(purchases.map(lambda x: x["user_lat"]).batch(100))

user_lat_buckets = np.linspace(
    np.concatenate(latitude).min(), 
    np.concatenate(latitude).max(), num=1000,
)

longitude = list(purchases.map(lambda x: x["user_long"]).batch(100))

user_long_buckets = np.linspace(
    np.concatenate(longitude).min(), 
    np.concatenate(longitude).max(), num=1000,
)

price = list(purchases.map(lambda x: x["price"]).batch(100))

price_buckets = np.linspace(
    np.concatenate(list(purchases.map(lambda x: x["price"]).batch(100))).min(), 
    np.concatenate(list(purchases.map(lambda x: x["price"]).batch(100))).max(), num=1000,
)


# In[115]:


class UserModel(tf.keras.Model):
  
    def __init__(self):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
        ])
        self.gender_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_gender, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_gender) + 1, 32),
        ])
        #self.timestamp_embedding = tf.keras.Sequential([
        #    tf.keras.layers.experimental.preprocessing.Discretization(timestamp_buckets.tolist()),
        #    tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
        #])
        #self.normalized_timestamp = tf.keras.layers.experimental.preprocessing.Normalization()

        #self.normalized_timestamp.adapt(timestamps)
        
        self.age_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Discretization(user_age_buckets.tolist()),
            tf.keras.layers.Embedding(len(user_age_buckets) + 1, 32),
        ])
        self.normalized_age = tf.keras.layers.experimental.preprocessing.Normalization()

        self.normalized_age.adapt(user_age_buckets)
        
        self.lat_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Discretization(user_lat_buckets.tolist()),
            tf.keras.layers.Embedding(len(user_lat_buckets) + 1, 32),
        ])
        self.normalized_lat = tf.keras.layers.experimental.preprocessing.Normalization()

        self.normalized_lat.adapt(user_lat_buckets)
        
        self.long_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Discretization(user_long_buckets.tolist()),
            tf.keras.layers.Embedding(len(user_long_buckets) + 1, 32),
        ])
        self.normalized_long = tf.keras.layers.experimental.preprocessing.Normalization()

        self.normalized_long.adapt(user_lat_buckets)


    def call(self, inputs):
        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.gender_embedding(inputs["user_gender"]),
            #self.timestamp_embedding(inputs["timestamp"]),
            #self.normalized_timestamp(inputs["timestamp"]),
            self.age_embedding(inputs["user_age"]),
            self.normalized_age(inputs["user_age"]),
            self.lat_embedding(inputs["user_lat"]),
            self.normalized_lat(inputs["user_lat"]),
            self.long_embedding(inputs["user_long"]),
            self.normalized_long(inputs["user_long"]),
        ], axis=1)


# In[116]:


class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes):
        """Model for encoding user queries.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = UserModel()

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))
    
    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


# In[117]:


class ProductModel(tf.keras.Model):
  
    def __init__(self):
        super().__init__()

        max_tokens = 10_000

        self.product_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_ids,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_ids) + 1, 32)
        ])

        self.product_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_text_embedding = tf.keras.Sequential([
          self.product_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.product_vectorizer.adapt(products)
        
        self.product_colour_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_colour,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_colour) + 1, 32)
        ])

        self.product_colour_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_colour_text_embedding = tf.keras.Sequential([
          self.product_colour_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.product_colour_vectorizer.adapt(purchases['product_colour'])
        
        self.product_tear_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_tear,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_tear) + 1, 32)
        ])

        self.product_tear_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_tear_text_embedding = tf.keras.Sequential([
          self.product_tear_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.product_tear_vectorizer.adapt(purchases['product_tear'])
        
        self.product_tonality_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_tonality,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_tonality) + 1, 32)
        ])

        self.product_tonality_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_tonality_text_embedding = tf.keras.Sequential([
          self.product_tonality_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.product_tonality_vectorizer.adapt(purchases['product_tonality'])
        
        self.product_gender_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_gender,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_gender) + 1, 32)
        ])

        self.product_gender_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_gender_text_embedding = tf.keras.Sequential([
          self.product_gender_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.product_gender_vectorizer.adapt(purchases['product_gender'])
        
        self.product_age_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_age,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_age) + 1, 32)
        ])

        self.product_age_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_age_text_embedding = tf.keras.Sequential([
          self.product_age_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.product_age_vectorizer.adapt(purchases['product_age'])
        
        self.product_category_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_category,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_category) + 1, 32)
        ])

        self.product_category_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_category_text_embedding = tf.keras.Sequential([
          self.product_category_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAvercategoryPooling1D(),
        ])

        self.product_category_vectorizer.adapt(purchases['product_category'])
        
        self.product_fit_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_fit,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_fit) + 1, 32)
        ])

        self.product_fit_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_fit_text_embedding = tf.keras.Sequential([
          self.product_fit_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAverfitPooling1D(),
        ])

        self.product_fit_vectorizer.adapt(purchases['product_fit'])
        
        self.product_rise_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_rise,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_rise) + 1, 32)
        ])

        self.product_rise_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_rise_text_embedding = tf.keras.Sequential([
          self.product_rise_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAverrisePooling1D(),
        ])

        self.product_rise_vectorizer.adapt(purchases['product_rise'])
        
        self.product_neckline_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_neckline,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_neckline) + 1, 32)
        ])

        self.product_neckline_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_neckline_text_embedding = tf.keras.Sequential([
          self.product_neckline_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAvernecklinePooling1D(),
        ])

        self.product_neckline_vectorizer.adapt(purchases['product_neckline'])
        
        self.product_sleeve_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_sleeve,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_sleeve) + 1, 32)
        ])

        self.product_sleeve_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_sleeve_text_embedding = tf.keras.Sequential([
          self.product_sleeve_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAversleevePooling1D(),
        ])

        self.product_sleeve_vectorizer.adapt(purchases['product_sleeve'])
        
        self.product_denim_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_denim,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_denim) + 1, 32)
        ])

        self.product_denim_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_denim_text_embedding = tf.keras.Sequential([
          self.product_denim_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAverdenimPooling1D(),
        ])

        self.product_denim_vectorizer.adapt(purchases['product_denim'])
        
        self.product_stretch_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_stretch,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_stretch) + 1, 32)
        ])

        self.product_stretch_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_stretch_text_embedding = tf.keras.Sequential([
          self.product_stretch_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAverstretchPooling1D(),
        ])

        self.product_stretch_vectorizer.adapt(purchases['product_stretch'])
        
        self.product_wash_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_product_wash,mask_token=None),
          tf.keras.layers.Embedding(len(unique_product_wash) + 1, 32)
        ])

        self.product_wash_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.product_wash_text_embedding = tf.keras.Sequential([
          self.product_wash_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAverwashPooling1D(),
        ])

        self.product_wash_vectorizer.adapt(purchases['product_wash'])
        
        self.price_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Discretization(price_buckets.tolist()),
            tf.keras.layers.Embedding(len(price_buckets) + 1, 32),
        ])
        self.normalized_price = tf.keras.layers.experimental.preprocessing.Normalization()

        self.normalized_age.adapt(price_buckets)

    def call(self, products):
        return tf.concat([
            self.product_embedding(inputs['product_id']),
            self.product_text_embedding(inputs['product_id']),
            self.product_colour_embedding(inputs['product_colour']),
            self.product_colour_text_embedding(inputs['product_colour']),
            self.product_tear_embedding(inputs['product_tear']),
            self.product_tear_text_embedding(inputs['product_tear']),
            self.product_tonality_embedding(inputs['product_tonality']),
            self.product_tonality_text_embedding(inputs['product_tonality']),
            self.product_gender_embedding(inputs['product_gender']),
            self.product_gender_text_embedding(inputs['product_gender']),
            self.product_age_embedding(inputs['product_age']),
            self.product_age_text_embedding(inputs['product_age']),
            self.product_category_embedding(inputs['product_category']),
            self.product_category_text_embedding(inputs['product_category']),
            self.product_category_embedding(inputs['product_category']),
            self.product_category_text_embedding(inputs['product_category']),
            self.product_fit_embedding(inputs['product_fit']),
            self.product_fit_text_embedding(inputs['product_fit']),
            self.product_rise_embedding(inputs['product_rise']),
            self.product_rise_text_embedding(inputs['product_rise']),
            self.product_neckline_embedding(inputs['product_neckline']),
            self.product_neckline_text_embedding(inputs['product_neckline']),
            self.product_sleeve_embedding(inputs['product_sleeve']),
            self.product_sleeve_text_embedding(inputs['product_sleeve']),
            self.product_denim_embedding(inputs['product_denim']),
            self.product_denim_text_embedding(inputs['product_denim']),
            self.product_stretch_embedding(inputs['product_stretch']),
            self.product_stretch_text_embedding(inputs['product_stretch']),
            self.product_wash_embedding(inputs['product_wash']),
            self.product_wash_text_embedding(inputs['product_wash']),
            self.price_embedding(inputs["price"]),
            self.normalized_price(inputs["price"]),
        ], axis=1)


# In[118]:


class CandidateModel(tf.keras.Model):
    """Model for encoding movies."""

    def __init__(self, layer_sizes):
        """Model for encoding movies.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = ProductModel()

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))
    
    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


# In[119]:


class MainModel(tfrs.models.Model):

    def __init__(self, layer_sizes):
        super().__init__()
        self.query_model = QueryModel(layer_sizes)
        self.candidate_model = CandidateModel(layer_sizes)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.candidate_model),
            ),
        )

        def compute_loss(self, features, training=False):
        # We only pass the user id and timestamp features into the query model. This
        # is to ensure that the training inputs would have the same keys as the
        # query inputs. Otherwise the discrepancy in input structure would cause an
        # error when loading the query model after saving it.
            query_embeddings = self.query_model({
                "user_id": features["user_id"],
                #"timestamp": features["timestamp"],
                "user_age": features["user_age"],
                "user_gender": features["user_gender"],
                "user_lat": features["user_lat"],
                "user_long": features["user_long"]
            })
            product_embeddings = self.candidate_model({
                "product_id": features["product_id"],
                "product_colour": features["product_colour"],
                "product_tear": features["product_tear"],
                "product_tonality": features["product_tonality"],
                "product_gender": features["product_gender"],
                "product_age": features["product_age"],
                "product_category": features["product_category"],
                "product_fit": features["product_fit"],
                "product_rise": features["product_rise"],
                "product_neckline": features["product_neckline"],
                "product_sleeve": features["product_sleeve"],
                "product_denim": features["product_denim"],
                "product_stretch": features["product_stretch"],
                "product_wash": features["product_wash"],
                "price":features["price"]
            })

            return self.task(
                query_embeddings, product_embeddings, compute_metrics=not training)


# In[120]:


# Randomly shuffle data and split between train and test.

tf.random.set_seed(42)
shuffled = purchases.shuffle(len(masterdf), seed=42, reshuffle_each_iteration=False)

train = shuffled.take(round(len(masterdf)*0.8))
test = shuffled.skip(round(len(masterdf)*0.8)).take(round(len(masterdf)*0.2))

cached_train = train.shuffle(len(masterdf)).batch(2048)
cached_test = test.batch(4096).cache()


# In[121]:


num_epochs = 3 # Less epoch to get to predictions quicker

model = MainModel([32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

one_layer_history = model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=5,
    epochs=num_epochs,
    verbose=0)

accuracy = one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")


# In[ ]:




