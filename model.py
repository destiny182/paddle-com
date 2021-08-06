import pgl
import math
import paddle.fluid as F
import paddle.fluid.layers as L
import pgl.layers.conv as conv
from transformer_gat_pgl import transformer_gat_pgl
from pgl.utils import paddle_helper
from pgl import message_passing
import math

def get_norm(indegree):
    float_degree = L.cast(indegree, dtype="float32")
    float_degree = L.clamp(float_degree, min=1.0)
    norm = L.pow(float_degree, factor=-0.5) 
    return norm
    
class res_unimp_large(object):
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 2)
        self.hidden_size = config.get("hidden_size", 128)
        self.out_size=config.get("out_size", 40)
        self.embed_size=config.get("embed_size", 100)
        self.heads = config.get("heads", 8) 
        self.dropout = config.get("dropout", 0.3)
        self.edge_dropout = config.get("edge_dropout", 0.0)
        self.use_label_e = config.get("use_label_e", False)
    
    # 编码输入        
    def embed_input(self, feature):   
        lay_norm_attr=F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
        lay_norm_bias=F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
        feature=L.layer_norm(feature, name='layer_norm_feature_input', 
                                      param_attr=lay_norm_attr, 
                                      bias_attr=lay_norm_bias)
        return feature
    
    # 连同部分已知的标签编码输入（MaskLabel）
    def label_embed_input(self, feature):
        label = F.data(name="label", shape=[None, 1], dtype="int64")
        label_idx = F.data(name='label_idx', shape=[None, 1], dtype="int64")

        label = L.reshape(label, shape=[-1])
        label_idx = L.reshape(label_idx, shape=[-1])

        embed_attr = F.ParamAttr(initializer=F.initializer.NormalInitializer(loc=0.0, scale=1.0))
        embed = F.embedding(input=label, size=(self.out_size, self.embed_size), param_attr=embed_attr )

        feature_label = L.gather(feature, label_idx, overwrite=False)
        feature_label = feature_label + embed
        feature = L.scatter(feature, label_idx, feature_label, overwrite=True)
     
        lay_norm_attr = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
        lay_norm_bias = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
        feature = L.layer_norm(feature, name='layer_norm_feature_input', 
                                      param_attr=lay_norm_attr, 
                                      bias_attr=lay_norm_bias)
        return feature
        
    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
            dropout = self.dropout
        else:
            edge_dropout = 0
            dropout = 0

        if self.use_label_e:
            feature = self.label_embed_input(feature)
        else:
            feature = self.embed_input(feature)
        if dropout > 0:
            feature = L.dropout(feature, dropout_prob=dropout, 
                                    dropout_implementation='upscale_in_train')
        
        #改变输入特征维度是为了Res连接可以直接相加
        feature = L.fc(feature, size=self.hidden_size * self.heads, name="init_feature")


        for i in range(self.num_layers - 1):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
            from model_unimp_large import graph_transformer, attn_appnp

            res_feature = feature

            feature, _, cks = graph_transformer(str(i), ngw, feature, 
                                             hidden_size=self.hidden_size,
                                             num_heads=self.heads, 
                                             concat=True, skip_feat=True,
                                             layer_norm=True, relu=True, gate=True)
            if dropout > 0:
                feature = L.dropout(feature, dropout_prob=dropout, 
                                     dropout_implementation='upscale_in_train') 
            
            # 下面这行便是Res连接了
            feature = res_feature + feature 
        
        feature, attn, cks = graph_transformer(str(self.num_layers - 1), ngw, feature, 
                                             hidden_size=self.out_size,
                                             num_heads=self.heads, 
                                             concat=False, skip_feat=True,
                                             layer_norm=False, relu=False, gate=True)

        feature = attn_appnp(ngw, feature, attn, alpha=0.2, k_hop=10)

        pred = L.fc(
            feature, self.num_class, act=None, name="pred_output")
        return pred