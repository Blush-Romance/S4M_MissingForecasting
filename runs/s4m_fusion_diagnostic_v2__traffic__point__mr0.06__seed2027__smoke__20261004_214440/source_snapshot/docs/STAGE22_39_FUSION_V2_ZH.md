# 步骤22.39 V2：原融合求值顺序修复

用户2026-10-04授权继续本阶段；范围仍是冻结checkpoint、同seed2027、Traffic
validation、batch16、192/96、1648窗、CRN-v1。主模型、数据、指标、容差均不改。

## V1失败证据及变更原因

21:38定位smoke显示，原型关闭与直接原_fuse_context(0,u,z)预测完全一致，
query/raw前后差均0；c-R的上下文最大差0.0001220703125，预测最大差
0.000396728515625，超过原1e-5。这是float32（32位浮点）分块相减与原
拼接GEMM（矩阵乘加）求值顺序不同，并非原型关闭失效或CRN失配。
V1配置/清单/runtime及两个失败run保留，不扩大容差。

## 结果前固定的V2算式

仍是同15条路径、同缩放比例。令F为原_fuse_context，不重编码、不重检索：

- 检索：F(a×r,u,z)；查询：F(r,a×u,z)，包含查询投影和残差。
- 历史：F(r,u,a×z)。
- 观测历史：F(r,u,where(mask,a×z,z))。
- 填补历史：F(r,u,where(mask,z,a×z))。

a=1直接返回原c，逐位保持baseline。上述与c+(a-1)×分量在实数下等价，
V2仅统一使用原融合的FP32求值路径。投影RMS/L2等统计仍按原分块定义测量。
旧c-R差值继续在smoke日志中留作数值定位，不作V2干预路径。
prototype-off门禁比较原型关闭与V2检索a=0，仍要求预测最大差≤1e-5。
CPU新增float64五分支多比例等价性、query残差、mask互补、identity、
非原地修改与直接零r严格一致检查，通过后才允许GPU。

## 版本和记录

独立configs/stage22_39_fusion_v2、runtime/stage22_39_fusion_v2、日志目录和
包含s4m_fusion_diagnostic_v2的全新实验ID；resolved_config明确求值顺序与原因。
保留V1冻结协议文档，V2文档优先描述本轮求值；不复用V1失败任务作成功结果。
实际文件/公式变化、CPU/GPU验证写入progress和badandgood简明记录。
完整科学边界沿用原协议：固定权重探针不是新模型结果，修复诊断不计作创新。
