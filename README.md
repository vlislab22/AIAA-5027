# AIAA 5027: Deep Learning for Visual Intelligence: Trends and Challenges 

## Course information

* Instructor: [[WANG Lin]](https://addisonwang2013.github.io/vlislab) (linwang@ust.hk)
* TAs: WU Hao (hwubx@connect.ust.hk), BAI Haotian (haotianbai@ust.hk) and CAO Zidong (zidongcao@ust.hk)
* Class time: Mon. 16.30 -17.50  & Fri. 10.30-11.50
* Office hours: BY appointment only.

## Course description

This is a task-oriented yet interaction-based course, which aims to scrutinize the recent trends and challenges of deep learning in visual intelligence tasks (learning methods, high- and low-level vision problems). This course will follow the way of flipped-classroom manner where the lecturer teaches the basics; meanwhile, the students will also be focused on active discussions, presentations (lecturing), and hands-on research projects under the guidance of the lecturer in the whole semester. Through this course, students will be equipped with the capability to critically challenge the existing methodologies/techniques and hopefully make breakthroughs in some new research directions.

## Grading policy 
- Paper summary (10%)
- Paper presentation and discussion (30%)
- Group project and paper submission (50%)
- Attendance and participation (10%)

##  Tentative schedule
| Dates | Topics | Active Learning |
| --- | --- | --- |
| 2/6 | Course introduction | |
| 2/10 | Course introduction | Overview of visual intelligence |
| 2/13 | Deep learning basics | TAs’ lectures for DL basics, algorithm basics and Pytorch tuorial  |
| 2/17 | Deep learning basics | TAs’ lectures for DL basics, algorithm basics and Pytorch tuorial  |
| 2/20 | DNN models in computer vision (VAE, GAN, Diffusion models) |   |
| 2/24 | DNN models in computer vision (VAE, GAN, Diffusion models) |  (1) Persentation (2) Review due 2/26 (3) Project meetings |
| 2/27 | Learning methods in computer vision (Transfer learning, domain adaptation, self/semi-supervised learning) |   |
| 3/3  | Learning methods in computer vision ((Transfer learning, domain adaptation, self/semi-supervised learning)) |  (1) Persentation (2) Review due 3/5  |
| 3/6  |Deep learning for image restoration and enhancement (I) deblurring, deraining, dehazing |   |
| 3/10 |Deep learning for image restoration and enhancement (I) deblurring, deraining, dehazing  |  (1) Persentation (2) Review due 3/12 (3) Project proposal kick-off (one page) |
| 3/13 |Deep learning for image restoration and enhancement (II) Image Super-resolution, HDR imaging |   |
| 3/17 |Deep learning for image restoration and enhancement (II) Image Super-resolution, HDR imaging |  (1) Persentation (2) Review due 3/19 |
| 3/20 |Deep learning for scene understanding (I) Object detection & tracking |   |
| 3/24 |Deep learning for scene understanding (I) Object detection & tracking | (1) Persentation (2) Review due 3/26  |
| 3/27 |Project mid-term presentation | |
| 3/31 |Project mid-term presentation | |
| 4/3 |Deep learning for scene understanding (II) Semantic segmentation  |  |
| 4/7 |Deep learning for scene understanding (II) Semantic segmentation  | (1) Persentation (2) Review due 4/12 |
| 4/10 |Depth and motion estimation (SLAM)  | |
| 4/14 |Depth and motion estimation (SLAM) | (1) Persenation (2) Review due 4/16 |
| 4/17 |Computer vision with novel cameras (I) Event camera-based vision  |  |
| 4/21 |Computer vision with novel cameras (I) Event camera-based vision  | (1) Persentation (2) Review due 4/19 |
| 4/24 |Computer vision with novel cameras (II) Thermal/360 camera-based vision  |  |
| 4/28 |Computer vision with novel cameras (II) Thermal/360 camera-based vision  | (1) Persentation (2) Review due 4/16 (3) Project meetings |
| 5/8  |Adversarial robustness in computer vision (Adversrial attack and defense) |  |
| 5/12 |Adversarial robustness in computer vision (Adversrial attack and defense)| (1) Persentation (2) Review due 4/30 (3) Project meetings |
| 5/19 |Project presentation and final paper submission |  |
| 5/22 |Project presentation and final paper submission | Submission due 5/30  |

---------------------------------------------------------------------------------
| 5/12 |Potential and challenges in visual intelligence (data, computation, learning, sensor) (NeRF for 3D reconstruction) |  |
| 5/15 |Potential and challenges in visual intelligence (data, computation, learning, sensor) (NeRF for 3D reconstruction)| (1) TA/Student lectures (2) final project Q/A  |
##  Reading list

### DNN models in computer vision (VAEs, GANs, Diffusion models)
#### VAEs 
[[Kingma and Welling 14]](https://arxiv.org/pdf/1312.6114v10.pdf) Auto-Encoding Variational Bayes, ICLR 2014. </br>
[[Kingma et al. 15]](https://arxiv.org/pdf/1506.02557.pdf) Variational Dropout and the Local Reparameterization Trick, NIPS 2015.</br>
[[Blundell et al. 15]](https://arxiv.org/pdf/1505.05424.pdf) Weight Uncertainty in Neural Networks, ICML 2015.</br>
[[Gal and Ghahramani 16]](http://proceedings.mlr.press/v48/gal16.pdf) Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning, ICML 2016. </br>

#### GANs
[[Goodfellow et al. 14] ](https://arxiv.org/pdf/1406.2661.pdf)Generative Adversarial Nets, NIPS 2014. </br>
[[Radford et al. 15] ](https://arxiv.org/pdf/1809.11096.pdf)Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016. </br>
[[Chen et al. 16]](https://arxiv.org/pdf/1606.03657.pdf) InfoGAN: Interpreting Representation Learning by Information Maximizing Generative Adversarial Nets, NIPS 2016. </br>
[[Arjovsky et al. 17]](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) Wasserstein Generative Adversarial Networks, ICML 2017. </br>
[[Zhu et al. 17]](https://arxiv.org/pdf/1703.10593.pdf) Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017.</br>
[[Liu et al. 17]](https://arxiv.org/pdf/1703.00848.pdf) UNIT: Unsupervised Image-to-Image Translation Networks, NeurIPS 2017. </br>
[[Choi et al. 18]](https://arxiv.org/pdf/1711.09020.pdf)StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation, CVPR 2018.  </br>
[[Isola et al. 17]](https://arxiv.org/pdf/1611.07004.pdf) Image-to-Image Translation with Conditional Adversarial Networks, CVPR, 2017. </br>
[[Huang et al. 17]](https://arxiv.org/pdf/1703.06868.pdf) Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization, ICCV, 2017. </br>
[[Huang et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.pdf)  Multimodal Unsupervised Image-to-Image Translation, ECCV, 2018. </br>

--- (**students' reading list**) ---

[[Brock et al. 19]](https://arxiv.org/pdf/1809.11096.pdf) Large Scale GAN Training for High-Fidelity Natural Image Synthesis, ICLR 2019. </br>
[[Karras et al. 19] ](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf) A Style-Based Generator Architecture for Generative Adversarial Networks, CVPR 2019. </br>
[[Karras et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf) Analyzing and Improving the Image Quality of StyleGAN, CVPR 2020. </br>
[[Park et al. 20] ](https://arxiv.org/pdf/2007.15651.pdf) Contrastive Learning for Unpaired Image-to-Image Translation, ECCV 2020. </br> 
[[Karras et al. 20]](https://arxiv.org/pdf/2006.06676.pdf) Training Generative Adversarial Networks with Limited Data, NeurIPS 2020. </br> 
[[Xie et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650494.pdf) Self-Supervised CycleGAN for Object-Preserving Image-to-Image Domain Adaptation, ECCV 2020. </br>
[[Mustafa et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630579.pdf) Transformation Consistency Regularization– A Semi-Supervised Paradigm for
Image-to-Image Translation, ECCV 2020. </br>
[[Li et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710647.pdf) Semantic Relation Preserving Knowledge Distillation for Image-to-Image Translation, ECCV, 2020.  </br>
[[Xu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Linear_Semantics_in_Generative_Adversarial_Networks_CVPR_2021_paper.pdf) Linear Semantics in Generative Adversarial Networks, CVPR, 2021.  </br>
[[Cao et al. 21]](https://arxiv.org/pdf/2103.16835.pdf) ReMix: Towards Image-to-Image Translation with Limited Data, CVPR 2021.  </br>
[[Liu et al. 21]](https://arxiv.org/pdf/2103.07893.pdf) DivCo: Diverse Conditional Image Synthesis via Contrastive Generative Adversarial Network, CVPR 2021.  </br>
[[Pizzati et al. 21]](https://arxiv.org/pdf/2103.06879.pdf) CoMoGAN: continuous model-guided image-to-image translation, CVPR 2021.  </br>
[[Jin et al. 21]](https://arxiv.org/pdf/2103.03467.pdf) Teachers Do More Than Teach: Compressing Image-to-Image Models, CVPR 2021.   </br>
[[Baek et al. 21]](https://arxiv.org/pdf/2006.06500.pdf) Rethinking the Truly Unsupervised Image-to-Image Translation, ICCV, 2021.   </br>
[[Wang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_TransferI2I_Transfer_Learning_for_Image-to-Image_Translation_From_Small_Datasets_ICCV_2021_paper.pdf) TransferI2I: Transfer Learning for Image-to-Image Translation from Small Datasets, ICCV, 2021. </br>
[[Yang et al. 21]](https://arxiv.org/pdf/2111.10346.pdf) Global and Local Alignment Networks for Unpaired Image-to-Image Translation, Arxiv 2021. </br>
[[Jiang et al. 21]](https://arxiv.org/pdf/2012.12821v3.pdf)  Focal Frequency Loss for Image Reconstruction and Synthesis, ICCV, 2021.  </br>
[[Zhang et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_StyleSwin_Transformer-Based_GAN_for_High-Resolution_Image_Generation_CVPR_2022_paper.pdf) StyleSwin: Transformer-based GAN for High-resolution Image Generation, CVPR, 2022.   </br>
[[Liao et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liao_Text_to_Image_Generation_With_Semantic-Spatial_Aware_GAN_CVPR_2022_paper.pdf) Text to Image Generation with Semantic-Spatial Aware GAN, CVPR, 2022.   </br>
[[Zhou et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Towards_Language-Free_Training_for_Text-to-Image_Generation_CVPR_2022_paper.pdf) Towards Language-Free Training for Text-to-Image Generation, CVPR, 2022.   </br>
[[Zhan et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhan_Marginal_Contrastive_Correspondence_for_Guided_Image_Generation_CVPR_2022_paper.pdf) Marginal Contrastive Correspondence for Guided Image Generation, CVPR, 2022.  </br>
[[Fruhstuck et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Fruhstuck_InsetGAN_for_Full-Body_Image_Generation_CVPR_2022_paper.pdf) InsetGAN for Full-Body Image Generation, CVPR, 2022.  </br>
[[He et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740626.pdf) PixelFolder: An Efficient Progressive Pixel Synthesis Network for Image Generation, ECCV, 2022.  </br>
[[Yang et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750036.pdf) WaveGAN: Frequency-Aware GAN for High-Fidelity Few-Shot Image Generation, ECCV, 2022.  </br>
[[Bai et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750036.pdf) High-fidelity GAN Inversion with Padding Space, ECCV, 2022.  </br>

#### DIffusion Model 
[[Sohl-Dickstein et al., 2015]](https://arxiv.org/abs/1503.03585)  Deep Unsupervised Learning using Nonequilibrium Thermodynamics, ICML , 2015. </br>
[[Song et al. 2019]](https://arxiv.org/pdf/1907.05600.pdf) Generative Modeling by Estimating Gradients of the Data Distribution, NeurIPS 2019.  </br>
[[Ho et al. 21]](https://arxiv.org/abs/2006.11239) Denoising Diffusion Probabilistic Models (DDPM), ICLR, 2021. </br>
[[Song et al. 21]](https://arxiv.org/pdf/2010.02502.pdf) Denoising Diffusion Implicit Models (DDIM), ICLR, 2021. </br>

--- (**students' reading list**) ---

[[Bao et al. 22]](https://arxiv.org/abs/2201.06503) "Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models, ICLR, 2022.  </br>
[[Openreview]](https://openreview.net/pdf?id=lsQCDXjOl3k) "Unconditional Diffusion Guidance, Openreview, 2022. </br>
[[Bao et al. 22]](https://arxiv.org/abs/2206.07309) Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models, ICML 2022. </br>
[[Dhariwal et al. 21]](https://arxiv.org/pdf/2105.05233.pdf) Diffusion Models Beat GANs on Image Synthesis, NIPS, 2021.   </br>
[[Meng et al. 22]](https://openreview.net/forum?id=aBsCjcPu_tE) SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations, ICLR, 2022. </br>
[[Song et al. 21]](https://arxiv.org/abs/2011.13456)  Score-based generative modeling through stochastic differential equations, ICLR, 2021. </br>
[[Nichol et al. 21]](https://arxiv.org/abs/2102.09672) Improved Denoising Diffusion Probabilistic Models, ICML, 2021.   </br>
[[Rombach et al. 22]](https://arxiv.org/abs/2112.10752) High-Resolution Image Synthesis with Latent Diffusion Models, CVPR, 2022.  </br>
[[Ho et al. 21]](https://arxiv.org/pdf/2106.15282.pdf) Cascaded Diffusion Models for High Fidelity Image Generation, Arxiv, 2021. </br>

### Learning methods in computer vision
#### Knowledge transfer 
[[Wang et al. 21]](https://arxiv.org/pdf/2004.05937v7.pdf) Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks, TPAMI, 2021. </br>
[[Hiton et al. 15]](https://arxiv.org/pdf/1503.02531.pdf) Distilling the Knowledge in a Neural Network, NIPS Workshop, 2015. </br>
[[Romero et al. 15]](https://arxiv.org/pdf/1412.6550.pdf) FitNets: Hints for Thin Deep Nets, ICLR, 2015. </br>
[[Gupta et al. 16]](https://arxiv.org/pdf/1507.00448.pdf) Cross Modal Distillation for Supervision Transfer, CVPR, 2016.    </br>
[[Zagoruyko et al. 16]](https://arxiv.org/pdf/1612.03928.pdf)  Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer, ICLR, 2017.  </br>
[[Furlanello et al. 18]](https://arxiv.org/abs/1805.04770) Born Again Neural Networks, ICML, 2018.   </br>
[[Zhang et al. 18]](https://arxiv.org/pdf/1706.00384.pdf) Deep Mutual Learning, CVPR,2018.  </br>
[[Tarvainen et al. 18]](https://arxiv.org/pdf/1703.01780.pdf)Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results, NIPS, 2018.  </br>
[[Zhang et al. 19]](https://arxiv.org/pdf/1905.08094.pdf)  Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation, ICCV, 2019.  </br> 
[[Heo et al. 19]](https://arxiv.org/pdf/1904.01866.pdf) A Comprehensive Overhaul of Feature Distillation, ICCV, 2019.  </br>
[[Tung et al.19]](https://arxiv.org/pdf/1907.09682.pdf) Similarity-Preserving Knowledge Distillation, ICCV, 2019.   </br>

--- Student's reading list-------------------- </br>
[[Chen et al. 19]](https://www.wangyunhe.site/data/2019%20ICCV%20DAFL.pdf) DAFL:Data-Free Learning of Student Networks, ICCV, 2019. </br>
[[Ahn et al. 19]](https://arxiv.org/pdf/1904.05835.pdf) Variational Information Distillation for Knowledge Transfer, CVPR, 2019.  </br>
[[Tian et al. 20]](https://arxiv.org/pdf/1910.10699v2.pdf)  Contrastive Representation Distillation, ICLR, 2020.  </br>
[[Fang et al. 20]](https://arxiv.org/pdf/1912.11006.pdf) Data-Free Adversarial Distillation, CVPR, 2020.  </br>
[[Yang et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460290.pdf) MutualNet: Adaptive ConvNet via Mutual Learning from Network Width and Resolution, ECCV, 2020. </br>
[[Yao et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600290.pdf) Knowledge Transfer via Dense Cross-layer Mutual-distillation. ECCV 2020 </br>
[[Guo et al. 20]](https://arxiv.org/pdf/2010.07485.pdf) Reducing the Teacher-Student Gap via Spherical Knowledge Disitllation, Arxiv, 2020.  </br>
[[Ji et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Refine_Myself_by_Teaching_Myself_Feature_Refinement_via_Self-Knowledge_Distillation_CVPR_2021_paper.pdf) Refine Myself by Teaching Myself: Feature Refinement via Self-Knowledge Distillation, CVPR, 2021. </br>
[[Liu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Source-Free_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2021_paper.pdf) Source-Free Domain Adaptation for Semantic Segmentation, CVPR, 2021. </br>
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Student_Networks_in_the_Wild_CVPR_2021_paper.pdf) Learning Student Networks in the Wild, CVPR, 2021. </br>
[[Xue et a. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xue_Multimodal_Knowledge_Expansion_ICCV_2021_paper.pdf) Multimodal Knowledge Expansion，ICCV, 2021. </br>
[[ZHu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhu_Student_Customized_Knowledge_Distillation_Bridging_the_Gap_Between_Student_and_ICCV_2021_paper.pdf) Student Customized Knowledge Distillation: Bridging the Gap Between Student and Teacher, ICCV,  2021. </br>
[[Kim et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Self-Knowledge_Distillation_With_Progressive_Refinement_of_Targets_ICCV_2021_paper.pdf) Self-Knowledge Distillation with Progressive Refinement of Targets, ICCV, 2021.   </br>
[[Son et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Son_Densely_Guided_Knowledge_Distillation_Using_Multiple_Teacher_Assistants_ICCV_2021_paper.pdf)  Densely Guided Knowledge Distillation using Multiple Teacher Assistants, ICCV, 2021.  </br>
[[Zhao et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.pdf)  Decoupled Knowledge Distillation, CVPR, 2022.  </br>
[[Chen et al. 22]](https://arxiv.org/pdf/2203.14001.pdf)  Knowledge Distillation with the Reused Teacher Classifier, CVPR, 2022.  </br>
[[Beyer et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.pdf)  Knowledge distillation: A good teacher is patient and consistent, CVPR, 2022.  </br>
[[Lin et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Knowledge_Distillation_via_the_Target-Aware_Transformer_CVPR_2022_paper.pdf)  Knowledge Distillation via the Target-aware Transformer, CVPR, 2022.  </br>
[[Yang et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840527.pdf)  MixSKD: Self-Knowledge Distillation from Mixup for Image Recognition, ECCV, 2022.  </br>
[[Shen et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840663.pdf) A Fast Knowledge Distillation Framework for Visual Recognition,ECCV,2022.  </br>
[[Shin et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720622.pdf) Teaching Where to Look: Attention Similarity Knowledge Distillation for Low Resolution Face Recognition, ECCV,2022.  </br>
[[Deng et al. 22]](https://link.springer.com/chapter/10.1007/978-3-031-19830-4_16) Personalized Education: Blind Knowledge Distillation, ECCV,2022.  </br>
[[Chen et al. 23]](https://openreview.net/pdf?id=8jU7wy7N7mA)  Supervision Complexity and its Role in Knowledge Distillation, ICLR, 2023.  </br>


#### Domain Adaptation
[[Long et al. 15] ](https://arxiv.org/pdf/1502.02791.pdf)Learning Transferable Features with Deep Adaptation Networks, ICML, 2015. </br>
[[Tzeng et al. 17]](https://arxiv.org/pdf/1702.05464.pdf) Adversarial Discriminative Domain Adaptation, CVPR, 2017. </br>
[[Huang et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Haoshuo_Huang_Domain_transfer_through_ECCV_2018_paper.pdf) Domain Transfer Through Deep Activation Matching, ECCV, 2018. </br>
[[Bermu’dez-Chaco’n et al. 20]](https://openreview.net/pdf?id=rJxycxHKDS) Domain Adaptive Multibranch Networks, ICLR, 2020. </br>
[[Carlucci et al. 17]](https://arxiv.org/pdf/1704.08082.pdf) AutoDIAL: Automatic DomaIn Alignment Layers, ICCV, 2017.  </br>
[[Chang et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)  Domain-Specific Batch Normalization for Unsupervised Domain Adaptation, CVPR, 2019. </br>
[[Cui et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cui_Towards_Discriminability_and_Diversity_Batch_Nuclear-Norm_Maximization_Under_Label_Insufficient_CVPR_2020_paper.pdf) Towards Discriminability and Diversity:Batch Nuclear-norm Maximization under Label Insufficient Situations, CVPR 2020. </br>
[[Roy et al. 19]](https://arxiv.org/pdf/1903.03215.pdf) Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss, CVPR, 2019. </br>
[[Csurka et al. 17]](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w38/Csurka_Discrepancy-Based_Networks_for_ICCV_2017_paper.pdf)  Discrepancy-based networks for unsupervised domain adaptation: a comparative study, CVPRW, 2017. </br>
[[Murez et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Murez_Image_to_Image_CVPR_2018_paper.pdf) Image to Image Translation for Domain Adaptation, CVPR, 2018. </br>
[[Liu et al. 17]](https://arxiv.org/pdf/1606.07536.pdf) Coupled Generative Adversarial Networks, NIPS, 2017. </br>
[[Hoffman et al. 18]](https://arxiv.org/pdf/1711.03213.pdf) CyCADA: Cycle-Consistent Adversarial Domain Adaptation, ICLR, 2018. </br>
[[Lee et al. 18]](https://arxiv.org/pdf/1808.00948.pdf) Diverse Image-to-Image Translation via Disentangled Representations, ECCV, 2018. </br>
[[Chen et al. 12]](https://arxiv.org/ftp/arxiv/papers/1206/1206.4683.pdf) Marginalized Denoising Autoencoders for Domain Adaptation, ICML, 2012. </br>
[[Zhuang et al. 15]](https://www.ijcai.org/Proceedings/15/Papers/578.pdf) Supervised Representation Learning: Transfer Learning with Deep Autoencoders, IJCAI, 2015. </br>
[[ Ghifary et al. 16]](https://arxiv.org/pdf/1607.03516.pdf) Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation, ECCV, 2016.  </br>
[[Bousmalis et al. 16]](https://arxiv.org/pdf/1608.06019.pdf) Domain Separation Networks, NIPS, 2016.  </br>
[[French et al. 19]](https://arxiv.org/pdf/1706.05208.pdf) Self-ensembling for Visual Domain Adaptation, ICLR, 2019.  </br>
[[Shu et al. 18]](https://arxiv.org/pdf/1802.08735.pdf) A DIRT-T Approach to Unsupervised Domain Adaptation, ICLR, 2018. </br>
[[ Deng et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Cluster_Alignment_With_a_Teacher_for_Unsupervised_Domain_Adaptation_ICCV_2019_paper.pdf) Cluster Alignment with a Teacher for Unsupervised Domain Adaptation, ICCV, 2019. </br>
[[Chen et al. 19]](https://arxiv.org/pdf/1811.08585.pdf) Progressive Feature Alignment for Unsupervised Domain Adaptation, CVPR 2019. </br>
[[Zhang et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1410.pdf)  Progressive Feature Alignment for Unsupervised Domain Adaptation, CVPR 2018. </br>
[[Kang et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)  Contrastive Adaptation Network for Unsupervised Domain Adaptation, CVPR 2019. </br>
 
------ Students' reading list----------------------------------------------</br>
[[Guizilini et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Guizilini_Geometric_Unsupervised_Domain_Adaptation_for_Semantic_Segmentation_ICCV_2021_paper.pdf)  Geometric Unsupervised Domain Adaptation for Semantic Segmentation, ICCV, 2021. </br>
[[Wang et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530715.pdf)  Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation, ECCV, 2020.  </br>
[[Peng et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510749.pdf) Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation, ECCV, 2020. </br>
[[Liu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Source-Free_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2021_paper.pdf)  Source-Free Domain Adaptation for Semantic Segmentation, CVPR, 2021. </br>
[[Na et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Na_FixBi_Bridging_Domain_Spaces_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf) FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation, CVPR, 2021.  </br>
[[Sharma et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Sharma_Instance_Level_Affinity-Based_Transfer_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf) Instance Level Affinity-Based Transfer for Unsupervised Domain Adaptation, CVPR, 2021.  </br>
[[Ahmed et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ahmed_Unsupervised_Multi-Source_Domain_Adaptation_Without_Access_to_Source_Data_CVPR_2021_paper.pdf) 
Unsupervised Multi-source Domain Adaptation Without Access to Source Data, CVPR, 2021. </br>
[[He et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/He_Multi-Source_Domain_Adaptation_With_Collaborative_Learning_for_Semantic_Segmentation_CVPR_2021_paper.pdf) Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation, CVPR, 2021. </br>
[[Wu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_DANNet_A_One-Stage_Domain_Adaptation_Network_for_Unsupervised_Nighttime_Semantic_CVPR_2021_paper.pdf) DANNet: A One-Stage Domain Adaptation Network for Unsupervised Nighttime Semantic Segmentation, CVPR, 2021. </br>
[[Lengyel et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Lengyel_Zero-Shot_Day-Night_Domain_Adaptation_With_a_Physics_Prior_ICCV_2021_paper.pdf) Zero-Shot Day-Night Domain Adaptation with a Physics Prior, ICCV, 2021. </br>
[[Li et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Semantic_Concentration_for_Domain_Adaptation_ICCV_2021_paper.pdf)  Semantic Concentration for Domain Adaptation, ICCV, 2021. </br>
[[Awais et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Awais_Adversarial_Robustness_for_Unsupervised_Domain_Adaptation_ICCV_2021_paper.pdf)  Adversarial Robustness for Unsupervised Domain Adaptation, ICCV, 2021.  </br>
[[Huang et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Category_Contrast_for_Unsupervised_Domain_Adaptation_in_Visual_Tasks_CVPR_2022_paper.pdf)  Category Contrast for Unsupervised Domain Adaptation in Visual Tasks, CVPR, 2022.  </br>
[[Wang et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Exploring_Domain-Invariant_Parameters_for_Source_Free_Domain_Adaptation_CVPR_2022_paper.pdf)  Exploring Domain-Invariant Parameters for Source Free Domain Adaptation, CVPR, 2022.</br>
[[Ding et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Source-Free_Domain_Adaptation_via_Distribution_Estimation_CVPR_2022_paper.pdf) Source-Free Domain Adaptation via Distribution Estimation, CVPR, 2022.</br>
[[Ding et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.pdf) Continual Test-Time Domain Adaptation, CVPR, 2022.</br>
[[Chen et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Reusing_the_Task-Specific_Classifier_as_a_Discriminator_Discriminator-Free_Adversarial_Domain_CVPR_2022_paper.pdf) Reusing the Task-specific Classifier as a Discriminator:
Discriminator-free Adversarial Domain Adaptation, CVPR, 2022.</br>
[[Oh et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810619.pdf) Reusing the Task-specific Classifier as a Discriminator:
Discriminator-free Adversarial Domain Adaptation, ECCV, 2022.</br>
[[Roy et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850530.pdf) Uncertainty-guided Source-free Domain Adaptation, ECCV, 2022.</br>
[[Lin et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930345.pdf) Prototype-Guided Continual Adaptation for Class-Incremental Unsupervised Domain Adaptation, ECCV, 2022.</br>
[[Lin et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930520.pdf) Adversarial Partial Domain Adaptation by Cycle Inconsistency, ECCV, 2022.</br>
[[Sun et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930628.pdf) Prior Knowledge Guided Unsupervised Domain Adaptationy, ECCV, 2022.</br>



#### Semi-supervised learning 
[[Sajjadi et al. 16]](https://arxiv.org/pdf/1606.04586.pdf) Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning, NIPS, 2016. </br>
[[Laine et al. 17]](https://arxiv.org/pdf/1610.02242.pdf)  Temporal Ensembling for Semi-Supervised Learning，ICLR, 2017. </br>
[[Tarvainen et al. 17]](https://arxiv.org/pdf/1703.01780.pdf) Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results, NIPS, 2017.  </br>
[[Miyato et al. 18]](https://arxiv.org/pdf/1704.03976.pdf) Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning, TPAMI, 2018.  </br>
[[Verma et al. 19]](https://arxiv.org/pdf/1903.03825.pdf) Interpolation Consistency Training for Semi-Supervised Learning, NIPS, 2019. </br>
[[Lee et al. 13]](url) Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks, ICML, 2013. </br>
[[Iscen et al. 19]](url) Label Propagation for Deep Semi-supervised Learning, CVPR, 2019.  </br>
[[Xie  et al. 20]](https://arxiv.org/pdf/1911.04252.pdf) Self-training with Noisy Student improves ImageNet classification, CVPR, 2020. </br>
[[Berthelot et al. 19]](https://arxiv.org/pdf/1905.02249.pdf) MixMatch: A Holistic Approach to Semi-Supervised Learning, NIPS, 2019. </br>
[[Berthelot et al. 20]](url) ReMixMatch: Semi-supervised learning with distribution alignment and augmentation anchoring, ICLR, 2020. </br>
[[Junnan Li et al. 20]](https://arxiv.org/pdf/2002.07394.pdf) DivideMix: Learning with Noisy Labels as Semi-supervised Learning, ICLR, 2020. </br>
[[Sohn et al. 20]](https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf) FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence, NIPS, 2020. </br>
[[Quali et al. 20]](https://arxiv.org/pdf/2006.05278.pdf)  An Overview of Deep Semi-Supervised Learning, 2020. </br>

----- Students' reading list-------------------</br>

[[Ke et al. 19]](https://arxiv.org/pdf/1909.01804.pdf) Dual Student: Breaking the Limits of the Teacher in Semi-supervised Learning, ICCV, 2019. </br>
[[Luo et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500766.pdf) Semi-supervised Semantic Segmentation via Strong-weak Dual-branch Network, ECCV, 2020. </br> 
[[Gao et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550511.pdf) Consistency-based Semi-supervised Active Learning: Towards Minimizing Labeling Cost, ECCV, 2020. </br>
[[Liu et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590307.pdf) Generative View-Correlation Adaptation for Semi-Supervised Multi-View Learning, ECCV, 2020. </br>
[[Kuo et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630460.pdf) FeatMatch: Feature-Based Augmentation for Semi-Supervised Learning, ECCV, 2020. </br>
[[Mustafa et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630579.pdf)  Transformation Consistency Regularization – A Semi-Supervised Paradigm for Image-to-Image Translation, ECCV, 2020. </br>
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Semi-Supervised_Semantic_Segmentation_With_Cross_Pseudo_Supervision_CVPR_2021_paper.pdf) Semi-Supervised Semantic Segmentation With Cross Pseudo Supervision,CVPR, 2021. </br>
[[Lai et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Lai_Semi-Supervised_Semantic_Segmentation_With_Directional_Context-Aware_Consistency_CVPR_2021_paper.pdf) Adaptive Consistency Regularization for Semi-Supervised Transfer Learning, CVPR,2021. </br>
[[Hu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_SimPLE_Similar_Pseudo_Label_Exploitation_for_Semi-Supervised_Classification_CVPR_2021_paper.pdf) SimPLE: Similar Pseudo Label Exploitation for Semi-Supervised Classification,CVPR,2021. </br>
 [[Zhou et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_C3-SemiSeg_Contrastive_Semi-Supervised_Segmentation_via_Cross-Set_Learning_and_Dynamic_Class-Balancing_ICCV_2021_paper.pdf) Pixel Contrastive-Consistent Semi-Supervised Semantic Segmentation, ICCV, 2021. </br>
[[Xiong et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiong_Multiview_Pseudo-Labeling_for_Semi-Supervised_Learning_From_Video_ICCV_2021_paper.pdf) Multiview Pseudo-Labeling for Semi-supervised Learning from Video, ICCV, 2021. </br>
[[Wang et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900423.pdf) Unsupervised Selective Labeling for More
Effective Semi-Supervised Learning, ECCV, 2022. </br>
[[Qin et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900477.pdf) Multi-Granularity Distillation Scheme Towards Lightweight Semi-Supervised Semantic Segmentation, ECCV, 2022. </br>
[[Weng et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900596.pdf) Semi-Supervised Vision Transformers, ECCV, 2022. </br>
[[Kim et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900665.pdf) ConMatch: Semi-Supervised Learning with Confidence-Guided Consistency Regularization, ECCV, 2022. </br>
[[Zheng et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_SimMatch_Semi-Supervised_Learning_With_Similarity_Matching_CVPR_2022_paper.pdf) SimMatch: Semi-supervised Learning with Similarity Matching, CVPR, 2022. </br>
[[Fan et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_CoSSL_Co-Learning_of_Representation_and_Classifier_for_Imbalanced_Semi-Supervised_Learning_CVPR_2022_paper.pdf) CoSSL: Co-Learning of Representation and Classifier for Imbalanced Semi-Supervised Learning, CVPR, 2022. </br>
[[Yang et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Class-Aware_Contrastive_Semi-Supervised_Learning_CVPR_2022_paper.pdf) Class-Aware Contrastive Semi-Supervised Learning, CVPR, 2022. </br>

### Image restoration and enhancement 

#### Image Deblurring 
[[Xu et al. 14]](https://papers.nips.cc/paper/2014/file/1c1d4df596d01da60385f0bb17a4a9e0-Paper.pdf) Deep Convolutional Neural Network for Image Deconvolution, NIPS, 2014. </br>
[[Zhang et al. 22]](https://arxiv.org/pdf/2201.10700.pdf) Deep Image Deblurring: A Survey, Arxiv, 2022. </br> 
[[Dong et al. 21]](https://arxiv.org/pdf/2103.09962.pdf) Deep Wiener Deconvolution: Wiener Meets Deep Learning for Image Deblurring, NIPS, 2021. </br> 
[[Nimisha et al., 17]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Nimisha_Blur-Invariant_Deep_Learning_ICCV_2017_paper.pdf) Blur-Invariant Deep Learning for Blind-Deblurring, ICCV, 2017. </br> 
[[Nah et al. 17]](https://arxiv.org/pdf/1612.02177.pdf) Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring, CVPR, 2017.  </br> 
[[Kupyn et al. 19]](https://arxiv.org/pdf/1908.03826.pdf) DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better, ICCV, 2019.  </br>
[[Zhang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Deblurring_by_Realistic_Blurring_CVPR_2020_paper.pdf) Deblurring by Realistic Blurring, CVPR, 2020.  </br> 
[[Zhou et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Spatio-Temporal_Filter_Adaptive_Network_for_Video_Deblurring_ICCV_2019_paper.pdf) Spatio-Temporal Filter Adaptive Network for Video Deblurring, ICCV, 2019.  </br> 
[[Nah et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nah_Recurrent_Neural_Networks_With_Intra-Frame_Iterations_for_Video_Deblurring_CVPR_2019_paper.pdf) Recurrent Neural Networks with Intra-Frame Iterations for Video Deblurring, CVPR, 2019.  </br> 
[[Purohit et al. 20]](https://arxiv.org/pdf/1903.11394.pdf) Region-Adaptive Dense Network for Efficient Motion Deblurring, AAAI,2020. (SoTA of single image deblur on GoPro dataset) </br> 
[[Shen et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shen_Human-Aware_Motion_Deblurring_ICCV_2019_paper.pdf)  Human-Aware Motion Deblurring, ICCV, 2019. </br> 

---- Students' reading list---------------------------------------

[[Rim et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700188.pdf)  Real-World Blur Dataset for Learning and Benchmarking Deblurring Algorithms, ECCV, 2020. </br> 
[[Lin et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530681.pdf) Learning Event-Driven Video Deblurring and Interpolation, ECCV, 2020.   </br> 
[[Zhong et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf)  Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring, ECCV, 2020.  </br> 
[[Abuolaim et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550120.pdf)  Defocus Deblurring Using Dual-Pixel Data, ECCV, 2020.  </br> 
[[Cun et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580732.pdf) Defocus Blur Detection via Depth Distillation, ECCV, 2020.   </br> 
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_a_Non-Blind_Deblurring_Network_for_Night_Blurry_Images_CVPR_2021_paper.pdf) Learning a Non-blind Deblurring Network for Night Blurry Images, CVPR, 2021.   </br> 
[[Rozumnyi et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Rozumnyi_DeFMO_Deblurring_and_Shape_Recovery_of_Fast_Moving_Objects_CVPR_2021_paper.pdf) DeFMO: Deblurring and Shape Recovery of Fast Moving Objects, CVPR, 2021.  </br> 
[[Xu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Motion_Deblurring_With_Real_Events_ICCV_2021_paper.pdf)  Motion Deblurring with Real Events, ICCV, 2021.  </br> 
[[Cho et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Cho_Rethinking_Coarse-To-Fine_Approach_in_Single_Image_Deblurring_ICCV_2021_paper.pdf) Rethinking Coarse-to-Fine Approach in Single Image Deblurring, ICCV, 2021.  </br> 
[[Shang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Shang_Bringing_Events_Into_Video_Deblurring_With_Non-Consecutively_Blurry_Frames_ICCV_2021_paper.pdf)  Bringing Events into Video Deblurring with Non-consecutively Blurry Frames, ICCV, 2021.   </br> 
[[Deng et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Deng_Multi-Scale_Separable_Network_for_Ultra-High-Definition_Video_Deblurring_ICCV_2021_paper.pdf) Multi-Scale Separable Network for Ultra-High-Definition Video Deblurring, ICCV, 2021.  </br> 
[[Hu et al 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_Pyramid_Architecture_Search_for_Real-Time_Image_Deblurring_ICCV_2021_paper.pdf)  Pyramid Architecture Search for Real-Time Image Deblurring, ICCV, 2021.  </br> 
[[Ji et al 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ji_XYDeblur_Divide_and_Conquer_for_Single_Image_Deblurring_CVPR_2022_paper.pdf)  XYDeblur: Divide and Conquer for Single Image Deblurring, CVPR, 2022.  </br> 
[[Whang et al 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Whang_Deblurring_via_Stochastic_Refinement_CVPR_2022_paper.pdf)  Deblurring via Stochastic Refinement, CVPR, 2022.  </br> 
[[Li et al 22]](https://arxiv.org/pdf/2208.05244.pdf) Learning Degradation Representations for Image Deblurring, ECCV, 2022.  </br> 
[[Tsai et al 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4651_ECCV_2022_paper.php) Stripformer: Strip Transformer for Fast Image Deblurring, ECCV, 2022.  </br> 
[[Zhong et al 22]](https://arxiv.org/pdf/2207.10123.pdf)  Animation from Blur: Multi-modal Blur Decomposition with Motion Guidance, ECCV, 2022. </br> 

#### Image deraining 
[[Li et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Single_Image_Deraining_A_Comprehensive_Benchmark_Analysis_CVPR_2019_paper.pdf) Single Image Deraining: A Comprehensive Benchmark Analysis, CVPR, 2019. </br>
[[Li  et al. 21]](https://link.springer.com/content/pdf/10.1007/s11263-020-01416-w.pdf) A Comprehensive Benchmark Analysis of Single Image Deraining:
Current Challenges and Future Perspectives, IJCV, 2021.   </br> 
[[Yang et al. 17]](https://arxiv.org/pdf/1609.07769.pdf) Deep Joint Rain Detection and Removal from a Single Image, CVPR, 2017.  </br> 
[[Zhang et al. 18]](https://arxiv.org/pdf/1802.07412.pdf) Density-aware Single Image De-raining using a Multi-stream Dense Network, CVPR, 2018.  </br> 
[[Hu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.pdf)   Depth-attentional features for single-image rain removal, CVPR, 2019.  </br> 
[[Qian et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qian_Attentive_Generative_Adversarial_CVPR_2018_paper.pdf)  Attentive Generative Adversarial Network for Raindrop Removal from A Single Image, CVPR, 2018.  </br> 
[[Zhang et al. 19]](https://arxiv.org/pdf/1701.05957.pdf)  Image de-raining using a conditional generative adversarial network, IEEE transactions on circuits and systems for video technology, 2019.  </br> 
[[Wei et al. 19]](https://arxiv.org/pdf/1807.11078.pdf) Semi-supervised Transfer Learning for Image Rain Removal, CVPR, 2019.  </br> 
[[Yang et al. 17]](url) Deep Joint Rain Detection and Removal from a Single Image, CVPR, 2017. </br> 
[[Hu et al. 17]](url) Depth-Attentional Features for Single-Image Rain Removal, CVPR, 2019. </br> 

--- Students' reading list-------------------------

[[Yasarla et al. 20]](https://arxiv.org/pdf/2006.05580.pdf)  Syn2Real Transfer Learning for Image Deraining using Gaussian Processes, CVPR, 2020.   </br> 
[[Liu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Unpaired_Learning_for_Deep_Image_Deraining_With_Rain_Direction_Regularizer_ICCV_2021_paper.pdf) Unpaired Learning for Deep Image Deraining with Rain Direction Regularizer, ICCV, 2021. </br> 
[[Zhou et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Image_De-Raining_via_Continual_Learning_CVPR_2021_paper.pdf)  Image De-raining via Continual Learning, CVPR, 2021.   </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Multi-Decoding_Deraining_Network_and_Quasi-Sparsity_Based_Training_CVPR_2021_paper.pdf) Multi-Decoding Deraining Network and Quasi-Sparsity Based Training, CVPR, 2021.  </br> 
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Robust_Representation_Learning_With_Feedback_for_Single_Image_Deraining_CVPR_2021_paper.pdf)  Robust Representation Learning with Feedback for Single Image Deraining, CVPR, 2021.  </br> 
[[Yue et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Semi-Supervised_Video_Deraining_With_Dynamical_Rain_Generator_CVPR_2021_paper.pdf)  Semi-Supervised Video Deraining with Dynamical Rain Generator, CVPR, 2021.   </br> 
[[Yi et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yi_Structure-Preserving_Deraining_With_Residue_Channel_Prior_Guidance_ICCV_2021_paper.pdf) Structure-Preserving Deraining with Residue Channel Prior Guidance, ICCV,2021.   </br> 
[[Huang et a. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Memory_Oriented_Transfer_Learning_for_Semi-Supervised_Image_Deraining_CVPR_2021_paper.pdf) Memory Oriented Transfer Learning for Semi-Supervised Image Deraining, CVPR, 2021.  </br> 
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper.pdf)  Pre-Trained Image Processing Transformer, CVPR, 2021.  </br> 
[[Jiang et al. 21]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_Multi-Scale_Progressive_Fusion_Network_for_Single_Image_Deraining_CVPR_2020_paper.pdf)  Multi-Scale Progressive Fusion Network for Single Image Deraining, CVPR, 2020.   </br> 
[[Fu et al. 20]](https://arxiv.org/pdf/1805.06173.pdf)  Lightweight Pyramid Networks for Image Deraining, IEEE Transactions on Neural Networks and Learning Systems, 2020. </br> 
[[Ba et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670713.pdf)  Not Just Streaks: Towards Ground Truth for Single Image Deraining, ECCV, 2022. </br> 
[[Ye et al. 22]](https://arxiv.org/pdf/2203.11509.pdf)  Unsupervised Deraining: Where Contrastive Learning Meets Self-similarity, CVPR, 2022. </br> 
[[Chen et al. 22]](https://arxiv.org/pdf/2109.02973.pdf) Unpaired Deep Image Deraining Using Dual Contrastive Learning, CVPR, 2022. </br> 

### Image dehazing
[[Gui et al. 21]](https://arxiv.org/pdf/2106.03323.pdf) A Comprehensive Survey on Image Dehazing Based on Deep Learning, IJCAI, 2021.  </br> 
[[Cai et al. 16]](https://arxiv.org/pdf/1601.07661.pdf) DehazeNet: An End-to-End System for Single Image Haze Removal, IEEE, TIP, 2016. </br> 
[[Ren et al. 20]](https://link.springer.com/content/pdf/10.1007/s11263-019-01235-8.pdf) Single Image Dehazing via Multi-scale Convolutional Neural Networks
with Holistic Edges, IJCV, 2020. (Extension of the conference version at 2016) </br> 
[[Li et al. 17]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_AOD-Net_All-In-One_Dehazing_ICCV_2017_paper.pdf) AOD-Net: All-in-One Dehazing Network, ICCV, 2017. </br> 
[[Qin et al. 20]](https://arxiv.org/pdf/1911.07559.pdf) FFA-Net: Feature Fusion Attention Network for Single Image Dehazing, AAAI,2020. </br> 
[[Zhang et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Densely_Connected_Pyramid_CVPR_2018_paper.pdf) Densely Connected Pyramid Dehazing Network, CVPR, 2018. </br> 
[[Ren et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ren_Gated_Fusion_Network_CVPR_2018_paper.pdf)  Gated Fusion Network for Single Image Dehazing
, CVPR, 2018. </br> 
[[Qu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qu_Enhanced_Pix2pix_Dehazing_Network_CVPR_2019_paper.pdf) Enhanced Pix2pix Dehazing Network, CVPR, 2019.  </br> 
[[Hong et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hong_Distilling_Image_Dehazing_With_Heterogeneous_Task_Imitation_CVPR_2020_paper.pdf)  Distilling Image Dehazing With Heterogeneous Task Imitation, CVPR, 2020.  </br> 
[[Shao et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shao_Domain_Adaptation_for_Image_Dehazing_CVPR_2020_paper.pdf) Domain Adaptation for Image Dehazing, CVPR, 2020.  </br> 
[[Engin et al. 18]]( https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Engin_Cycle-Dehaze_Enhanced_CycleGAN_CVPR_2018_paper.pdf )Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing, ECCVW, 2018. </br> 
[[Li et al. 20]](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/stamp/stamp.jsp?tp=&arnumber=9170880&tag=1) Zero-Shot Image Dehazing, IEEE TIP, 2020.  </br> 

--- Students' reading list-----------------------------

[[Wu et al. 21]](https://arxiv.org/pdf/2104.09367.pdf) Contrastive Learning for Compact Single Image Dehazing, CVPR, 2021. </br> 
[[Shyam et al. 21]](https://arxiv.org/pdf/2101.10449.pdf) Towards Domain Invariant Single Image Dehazing, AAAI, 2021. </br> 
[[Zheng et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Ultra-High-Definition_Image_Dehazing_via_Multi-Guided_Bilateral_Learning_CVPR_2021_paper.pdf) Ultra-High-Defifinition Image Dehazing via Multi-Guided Bilateral Learning, CVPR, 2021. </br> 
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_PSD_Principled_Synthetic-to-Real_Dehazing_Guided_by_Physical_Priors_CVPR_2021_paper.pdf) PSD: Principled Synthetic-to-Real Dehazing Guided by Physical Priors, CVPR, 2021. </br> 
[[Zhao et al. 21]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_BidNet_Binocular_Image_Dehazing_Without_Explicit_Disparity_Estimation_CVPR_2020_paper.pdf) BidNet: Binocular Image Dehazing without Explicit Disparity Estimation, CVPR, 2021.   </br> 
[[Kar et al. 21]](http://xxx.itp.ac.cn/pdf/2008.01701v1) Transmission Map and Atmospheric Light Guided Iterative Updater Network for Single Image Dehazing, CVPR, 2021. </br> 
[[Li et al. 20]](https://ieeexplore.ieee.org/abstract/document/8902220) Semi-Supervised Image Dehazing, IEEE TIP, 2020.  </br> 
[[Yi et al. 21]](https://arxiv.org/pdf/2102.03501.pdf)  Two-Step Image Dehazing with Intra-domain and Inter-domain Adaptation, Arxiv, 2021.</br> 
[[Liu et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Towards_Multi-Domain_Single_Image_Dehazing_via_Test-Time_Training_CVPR_2022_paper.pdf)  Towards Multi-Domain Single Image Dehazing via Test-Time Training, CVPR, 2022. </br> 
[[Yang et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Self-Augmented_Unpaired_Image_Dehazing_via_Density_and_Depth_Decomposition_CVPR_2022_paper.pdf) Self-augmented Unpaired Image Dehazing via Density and Depth Decomposition, CVPR, 2022. </br> 
[[Chen et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770636.pdf) Unpaired Deep Image Dehazing Using
Contrastive Disentanglement Learning, ECCV, 2022. </br> 

####  Image/Video Super-Resolution 
[[Dong et al. 16]](https://arxiv.org/pdf/1501.00092.pdf) mage Super-Resolution Using Deep Convolutional Networks, ECCV,2016.(First deep learning-based method)  </br>
[[Lim et al. 17]](https://arxiv.org/pdf/1707.02921.pdf) Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW, 2017.  </br> 
[[Wang et al. 19]](https://arxiv.org/pdf/1902.06068.pdf)  Deep Learning for Image Super-resolution: A Survey, IEEE TPAMI, 2021. </br> 
[[Kim et al. 17]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.pdf) Accurate Image Super-Resolution Using Very Deep Convolutional Networks, CVPR, 2017.  </br> 
[[Tai et al. 17]](https://arxiv.org/pdf/1708.02209.pdf) MemNet: A Persistent Memory Network for Image Restoration, CVPR, 2017.  </br> 
[[Li et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf)  Multi-scale Residual Network for Image Super-Resolution, ECCV, 2018. </br> 
[[Zhang et al. 18]](https://arxiv.org/pdf/1807.02758.pdf) Image Super-Resolution Using Very Deep Residual Channel Attention Networks, ECCV, 2018. </br> 
[[Zhang et al. 19]](https://arxiv.org/pdf/1903.10082.pdf) Residual Non-local Attention Networks for Image Restoration, ICLR, 2019. </br> 
[[Dai et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf) Second-order Attention Network for Single Image Super-Resolution, CVPR, 2019. </br> 
[[Han et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3225.pdf)  Image Super-Resolution via Dual-State Recurrent Networks, CVPR, 2018. </br> 
[[Li et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf) Multi-scale Residual Network for Image Super-Resolution, ECCV, 2018. </br> 
[[Ren et al. 18]](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Ren_Image_Super_Resolution_CVPR_2017_paper.pdf) Image Super Resolution Based on Fusing Multiple Convolution Neural Networks, CVPRW, 2017. </br> 
[[Ahn et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Namhyuk_Ahn_Fast_Accurate_and_ECCV_2018_paper.pdf) Fast, accurate, and lightweight
super-resolution with cascading residual network, ECCV, 2018. </br> 
[[Zhang et al. 19]](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/stamp/stamp.jsp?tp=&arnumber=8502129&tag=1)  DCSR: Dilated Convolutions for Single Image Super-Resolution, IEEE TIP, 2019. </br> 
[[Zhantg et al. 18]](https://arxiv.org/pdf/1802.08797.pdf) Residual Dense Network for Image Super-Resolution, CVPR, 2018. </br> 
[[Hu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Meta-SR_A_Magnification-Arbitrary_Network_for_Super-Resolution_CVPR_2019_paper.pdf) Meta-SR: A Magnification-Arbitrary Network for Super-Resolution, CVPR, 2021. </br> 
[[Chen et al. 21]](https://arxiv.org/pdf/2012.09161.pdf)  Learning Continuous Image Representation with Local Implicit Image Function, CVPR, 2021. </br> 
[[Lee et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690460.pdf) Learning with Privileged Information for Efficient Image Super-Resolution, ECCV, 2020. </br> 
[[Hu et al. 21]](https://www.ijcai.org/proceedings/2021/0155.pdf) Towards Compact Single Image Super-Resolution via Contrastive Self-distillation, IJCAI, 2021.  </br> 
[[Cai et al. 19]](https://csjcai.github.io/papers/RealSR.pdf) Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model, ICCV, 2019. </br> 
[[Wei et al. 20]](https://arxiv.org/pdf/2008.01928.pdf) Component Divide-and-Conquer for Real-World Image Super-Resolution, ECCV, 2021. </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Unsupervised_Real-World_Super-Resolution_A_Domain_Adaptation_Perspective_ICCV_2021_paper.pdf) Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective, ICCV, 2021. </br> 
[[Maeda et a. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Maeda_Unpaired_Image_Super-Resolution_Using_Pseudo-Supervision_CVPR_2020_paper.pdf) Unpaired Image Super-Resolution using Pseudo-Supervision, CVPR, 2020. </br> 
[[Shocher et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.pdf) “Zero-Shot” Super-Resolution using Deep Internal Learning, CVPR, 2018. </br> 

------- Students' reading list---------------------------------------------------

[[Wei et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wei_Unsupervised_Real-World_Image_Super_Resolution_via_Domain-Distance_Aware_Training_CVPR_2021_paper.pdf) Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective, ICCV, 2021. </br> 
[[Zhang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Data-Free_Knowledge_Distillation_for_Image_Super-Resolution_CVPR_2021_paper.pdf) Unsupervised Real-world Image Super Resolution via Domain-distance Aware Training, CVPR, 2021.   </br> 
[[Sefi et al. 20]](https://arxiv.org/pdf/1909.06581.pdf) Blind Super-Resolution Kernel Estimation using an Internal-GAN, NIPS, 2020.  </br> 
[[Cheng et a. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620256.pdf)  Zero-Shot Image Super-Resolution with Depth Guided Internal Degradation Learning, ECCV, 2020.  </br> 
[[Sun et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Learning_Scene_Structure_Guidance_via_Cross-Task_Knowledge_Transfer_for_Single_CVPR_2021_paper.pdf) Learning Scene Structure Guidance via Cross-Task Knowledge Transfer for Single Depth Super-Resolution, CVPR, 2021.  </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Unsupervised_Degradation_Representation_Learning_for_Blind_Super-Resolution_CVPR_2021_paper.pdf) Unsupervised Degradation Representation Learning for Blind Super-Resolution, CVPR, 2021.  </br> 
[[Son et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Son_SRWarp_Generalized_Image_Super-Resolution_under_Arbitrary_Transformation_CVPR_2021_paper.pdf) SRWarp: Generalized Image Super-Resolution under Arbitrary Transformation, CVPR, 2021.  </br> 
[[Jo et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Jo_Tackling_the_Ill-Posedness_of_Super-Resolution_Through_Adaptive_Target_Generation_CVPR_2021_paper.pdf) Tackling the Ill-Posedness of Super-Resolution through Adaptive Target Generation, CVPR, 2021.  </br> 
[[Mei et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Image_Super-Resolution_With_Non-Local_Sparse_Attention_CVPR_2021_paper.pdf) Image Super-Resolution with Non-Local Sparse Attention, CVPR, 2021.  </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_a_Single_Network_for_Scale-Arbitrary_Super-Resolution_ICCV_2021_paper.pdf) Learning a Single Network for Scale-Arbitrary Super-Resolution, ICCV, 2021.  </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Dual-Camera_Super-Resolution_With_Aligned_Attention_Modules_ICCV_2021_paper.pdf) Dual-Camera Super-Resolution with Aligned Attention Modules, CVPR, 2021. </br> 
[[Chan et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chan_BasicVSR_The_Search_for_Essential_Components_in_Video_Super-Resolution_and_CVPR_2021_paper.pdf) BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond, ICCV, 2021. </br> 
[[Yi et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yi_Omniscient_Video_Super-Resolution_ICCV_2021_paper.pdf)  Omniscient Video Super-Resolution, ICCV, 2021. </br> 
[[Tian et al. 20]](https://arxiv.org/pdf/1812.02898.pdf) TDAN: Temporally Deformable Alignment Network for Video Super-Resolution, CVPR, 2020. </br> 
[[Wang et al. 19]](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.pdf)  EDVR: Video Restoration With Enhanced Deformable Convolutional Networks, CVPRW, 2019. </br> 
[[Guo et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_LAR-SR_A_Local_Autoregressive_Model_for_Image_Super-Resolution_CVPR_2022_paper.pdf)  LAR-SR: A Local Autoregressive Model for Image Super-Resolution, CVPR, 2022. </br> 
[[Yue et al. 22]](https://arxiv.org/abs/2107.00986)  Blind Image Super-resolution with Elaborate Degradation Modeling on Noise and Kernel, CVPR, 2022. </br> 
[[Guo et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_LAR-SR_A_Local_Autoregressive_Model_for_Image_Super-Resolution_CVPR_2022_paper.pdf)  LAR-SR: A Local Autoregressive Model for Image Super-Resolution, CVPR, 2022. </br> 
[[Xu et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Dual_Adversarial_Adaptation_for_Cross-Device_Real-World_Image_Super-Resolution_CVPR_2022_paper.pdf)  Dual Adversarial Adaptation for Cross-Device Real-World Image Super-Resolution, CVPR, 2022. </br> 
[[Cao et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780318.pdf)  Reference-based Image Super-Resolution with Deformable Attention Transformer, ECCV, 2022. </br> 
[[Li et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780234.pdf)  MuLUT: Cooperating Multiple Look-Up Tables for Efficient Image Super-Resolution, ECCV, 2022. </br> 
 
#### Deep HDR imaging 

[[Wang et al. 21]](https://arxiv.org/pdf/2110.10394.pdf)  Deep Learning for HDR Imaging:State-of-the-Art and Future Trends, IEEE TPAMI, 2021. </br> 
[[Kalantrai et al. 17]](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR.pdf) Deep High Dynamic Range Imaging of Dynamic Scenes, Siggraph, 2017. </br> 
[[Prabhakar et al. 19]](https://ieeexplore.ieee.org/document/8747329) A Fast, Scalable, and Reliable Deghosting Method for Extreme Exposure Fusion, ICCP, 2019. </br> 
[[Wu et al. 18]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Shangzhe_Wu_Deep_High_Dynamic_ECCV_2018_paper.pdf) Deep High Dynamic Range Imaging with Large Foreground Motions, ECCV, 2018. </br> 
[[Yan et al. 21]](https://www.sciencedirect.com/science/article/abs/pii/S092523122031849X) Towards accurate HDR imaging with learning generator constraints, Neurocomputing, 2020. </br> 
[[Yan et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yan_Attention-Guided_Network_for_Ghost-Free_High_Dynamic_Range_Imaging_CVPR_2019_paper.pdf)  Attention-guided Network for Ghost-free High Dynamic Range Imaging, CVPR, 2019. </br> 
[[Rosh et al. 19]](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/stamp/stamp.jsp?tp=&arnumber=8803582) Deep Multi-Stage Learning for HDR With Large Object Motions, ICCP, 2019. </br> 
[[Xu et al. 20]](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/stamp/stamp.jsp?tp=&arnumber=9112609) MEF-GAN: Multi-Exposure Image Fusion via Generative Adversarial Networks, TIP, 2020. </br> 
[[Eilertsen et al. 17]](https://arxiv.org/pdf/1710.07480.pdf)  HDR image reconstruction from a single exposure using deep CNNs, Siggraph, 2017. </br> 
[[Santas et al. 20]](https://arxiv.org/pdf/2005.07335.pdf) Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss, Siggraph, 2020. </br> 
[[Endo et al. 17]](http://www.cgg.cs.tsukuba.ac.jp/~endo/projects/DrTMO/paper/DrTMO_SIGGRAPHAsia_light.pdf) Deep Reverse Tone Mapping, Siggraph, 2017. </br> 
[[Liu et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Single-Image_HDR_Reconstruction_by_Learning_to_Reverse_the_Camera_Pipeline_CVPR_2020_paper.pdf) Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline, CVPR, 2020. </br> 

-------Student's reading list----------------------------

[[Metzler]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Metzler_Deep_Optics_for_Single-Shot_High-Dynamic-Range_Imaging_CVPR_2020_paper.pdf) Deep Optics for Single-shot High-dynamic-range Imaging, CVPR, 2020. </br> 
[[Kim et al. 18]](https://link.springer.com/content/pdf/10.1007%2F978-3-030-20893-6_24.pdf) A Multi-purpose Convolutional Neural Network for Simultaneous Super-Resolution and High Dynamic Range Image Reconstruction, ACCV, 2018. </br> 
[[Kim et al. 19]](https://arxiv.org/ftp/arxiv/papers/1904/1904.11176.pdf) Deep sr-itm: Joint learning of superresolution and inverse tone-mapping for 4k uhd hdr applications, ICCV,2019. </br> 
[[Kim et al. 20]](https://arxiv.org/pdf/1909.04391.pdf)  JSI-GAN: GAN-Based Joint Super-Resolution and Inverse Tone-Mapping with Pixel-Wise Task-Specific Filters for UHD HDR Video, AAAI, 2020. </br> 
[[Kim et al. 20]](https://arxiv.org/pdf/2006.15833.pdf) End-to-End Differentiable Learning to HDR Image Synthesis for Multi-exposure Images, AAAI, 2020. </br> 
[[Chen et al. 21]](https://arxiv.org/pdf/2103.14943.pdf)  HDR Video Reconstruction: A Coarse-to-fine Network and A Real-world Benchmark Dataset, ICCV, 2021. </br> 
[[Jiang et al. 21]](https://arxiv.org/pdf/2103.10982.pdf) HDR Video Reconstruction with Tri-Exposure Quad-Bayer Sensors, Arxiv, 2021. </br> 
[[Mildenhall et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Mildenhall_NeRF_in_the_Dark_High_Dynamic_Range_View_Synthesis_From_CVPR_2022_paper.pdf)  NeRF in the Dark: High Dynamic Range View Synthesis From Noisy Raw Images, CVPR, 2022.  </br> 
[[Huang et al. 22]](https://arxiv.org/abs/2111.14451) HDR-NeRF: High Dynamic Range Neural Radiance Fields, CVPR, 2022.  </br> 
[[Vien et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670429.pdf) Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging, ECCV, 2022.  </br> 



### Object detection

[[Wu et al. 20]](https://arxiv.org/pdf/1908.03673.pdf) Recent advances in deep learning for object detection, Neurocomputing, 2020. </br> 
[[Girshick et al. 15]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) Fast R-CNN, ICCV, 2015.  </br> 
[[Ghodrati et al. 15]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ghodrati_DeepProposal_Hunting_Objects_ICCV_2015_paper.pdf) DeepProposal: Hunting Objects by Cascading Deep Convolutional Layers, ICCV, 2015. </br> 
[[Ren et al. 15]](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)  Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, NIPS, 2016. </br> 
[[Kong et al. 16]](https://zpascal.net/cvpr2016/Kong_HyperNet_Towards_Accurate_CVPR_2016_paper.pdf)  HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection, CVPR, 2016.  </br> 
[[He et al. 14]](https://arxiv.org/pdf/1406.4729.pdf) Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, ECCV, 2014. </br> 
[[Cai et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf)  Cascade R-CNN: Delving into High Quality Object Detection, CVPR, 2018. </br> 
[[Redmon et al. 16]](https://arxiv.org/pdf/1506.02640.pdf) You Only Look Once: Unified, Real-Time Object Detection, CVPR, 2016.  </br> 
[[Liu et al. 16]](https://arxiv.org/pdf/1512.02325.pdf) SSD: Single Shot MultiBox Detector, ECCV, 2016. </br> 
[[Lin et al. 18]](https://arxiv.org/pdf/1708.02002.pdf) Focal Loss for Dense Object Detection (RetinaNet), CVPR, 2018. </br> 
[[Redmon et al. 16]](https://arxiv.org/pdf/1612.08242.pdf) YOLO9000: Better, Faster, Stronger, Arxiv, 2017. </br> 
[[Law et al. 19]](https://arxiv.org/pdf/1808.01244.pdf)  CornerNet: Detecting Objects as Paired Keypoints,IJCV, 2019. </br> 
[[He et al. 15]](https://arxiv.org/pdf/1406.4729.pdf)  Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, IEEE TPAMI, 2015.  </br> 
[[Long et al. 16]](https://arxiv.org/pdf/1605.06409.pdf) R-FCN: Object Detection via Region-based Fully Convolutional Networks, NIPS, 2016. </br> 
[[Lin et al. 17]](https://arxiv.org/pdf/1612.03144.pdf) Feature Pyramid Networks for Object Detection, CVPR, 2017.  </br> 
[[He et al. 18]](url) Mask R-CNN, ICCV, 2018.  </br> 
[[Chen et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Towards_Accurate_One-Stage_Object_Detection_With_AP-Loss_CVPR_2019_paper.pdf)  Towards Accurate One-Stage Object Detection with AP-Loss, CVPR, 2019.  </br> 


---------- Student's reading list----------

###Generic detection

[[Redmon et al. 18]](https://arxiv.org/pdf/1804.02767.pdf) YOLOv3: An Incremental Improvement, Arxiv, 2018. </br> 
[[Chen et al. 19]](https://proceedings.neurips.cc/paper/2017/file/e1e32e235eee1f970470a3a6658dfdd5-Paper.pdf)  Learning Efficient Object Detection Models with Knowledge Distillation, NIPS, 2019. </br> 
[[Kang et al. 21]](https://papers.nips.cc/paper/2021/file/892c91e0a653ba19df81a90f89d99bcd-Paper.pdf)  Instance-Conditional Knowledge Distillation for Object Detection, NIPS, 2021. </br> 
[[Fang et al. 21]](https://arxiv.org/pdf/2106.00666.pdf) You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection, NIPS, 2021. </br> 
[[Ge et al. 21]](https://arxiv.org/pdf/2107.08430.pdf)  YOLOX: Exceeding YOLO Series in 2021, Arxiv, 2021. </br> 
[[Pramanik et al. 22]](https://ieeexplore.ieee.org/document/9313052) Granulated RCNN and Multi-Class Deep SORT for Multi-Object Detection and Tracking, IEEE TETCI, 2022. </br> 
[[Wang et al. 21]](https://arxiv.org/pdf/2105.04206.pdf)  You Only Learn One Representation: Unified Network for Multiple Tasks, Arxiv, 2021. </br> 
[[Wang et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Towards_Universal_Object_Detection_by_Domain_Attention_CVPR_2019_paper.pdf) Towards Universal Object Detection by Domain Attention, CVPR, 2019. </br> 
[[Huang et al. 19]](https://arxiv.org/pdf/1903.00241.pdf) Mask Scoring R-CNN, CVPR, 2019. </br> 
[[Guo et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Distilling_Object_Detectors_via_Decoupled_Features_CVPR_2021_paper.pdf) Distilling Object Detectors via Decoupled Features, CVPR, 2021. </br> 
[[Chen et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf) Domain Adaptive Faster R-CNN for Object Detection in the Wild, CVPR, 2018. </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Data-Uncertainty_Guided_Multi-Phase_Learning_for_Semi-Supervised_Object_Detection_CVPR_2021_paper.pdf) Data-Uncertainty Guided Multi-Phase Learning for Semi-Supervised Object Detection, CVPR,2021.  </br> 
[[Zhou et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Instant-Teaching_An_End-to-End_Semi-Supervised_Object_Detection_Framework_CVPR_2021_paper.pdf) Instant-Teaching: An End-to-End Semi-Supervised Object Detection Framework, CVPR, 2021.  </br> 
[[Yang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Interactive_Self-Training_With_Mean_Teachers_for_Semi-Supervised_Object_Detection_CVPR_2021_paper.pdf) Interactive Self-Training with Mean Teachers for Semi-supervised Object Detection, CVPR, 2021. </br> 
[[Wang et al. 23]](https://arxiv.org/pdf/2207.02696.pdf) YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors, CVPR, 2023. </br>
[[Feng et al. 22]](https://arxiv.org/abs/2205.04072) Beyond Bounding Box: Multimodal Knowledge Learning for Object Detection, CVPR, 2022. </br> 
[[Li et al. 22]](https://arxiv.org/abs/2111.13216) Cross-Domain Adaptive Teacher for Object Detection, CVPR, 2022. </br> 
[[Yang et al. 22]](https://arxiv.org/abs/2111.11837) Focal and Global Knowledge Distillation for Detectors, CVPR, 2022. </br> 

---
#### Face detection 
[[Luo et al. 16]](https://proceedings.neurips.cc/paper/2016/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf) Understanding the Effective Receptive Field in Deep Convolutional Neural Networks, 2016. </br> 
[[Tang et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xu_Tang_PyramidBox_A_Context-assisted_ECCV_2018_paper.pdf) PyramidBox: A Context-assisted Single Shot Face Detector, ECCV, 2018. </br> 
[[Liu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.pdf) High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection, CVPR, 2019. </br> 
[[Li et al. 20]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_DSFD_Dual_Shot_Face_Detector_CVPR_2019_paper.pdf) Dsfd: Dual shot face detector， CVPR, 2019. </br> 
[[Wang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Hierarchical_Pyramid_Diverse_Attention_Networks_for_Face_Recognition_CVPR_2020_paper.pdf) Hierarchical Pyramid Diverse Attention Networks for Face Recognition, CVPR, 2020. </br> 
[[Huang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_When_Age-Invariant_Face_Recognition_Meets_Face_Age_Synthesis_A_Multi-Task_CVPR_2021_paper.pdf) When Age-Invariant Face Recognition Meets Face Age Synthesis: A Multi-Task Learning Framework, CVPR, 2021. </br> 
[[Tong et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Tong_FaceSec_A_Fine-Grained_Robustness_Evaluation_Framework_for_Face_Recognition_Systems_CVPR_2021_paper.pdf) FACESEC: A Fine-grained Robustness Evaluation Framework for Face Recognition Systems, CVPR, 2021. </br> 
[[Qiu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiu_SynFace_Face_Recognition_With_Synthetic_Data_ICCV_2021_paper.pdf) SynFace: Face Recognition with Synthetic Data, ICCV, 2021. </br> 
[[Song et al. 21]](https://openaccess.thecvf.com/content/ICCV2021W/MFR/papers/Huang_Masked_Face_Recognition_Datasets_and_Validation_ICCVW_2021_paper.pdf)  Occlusion Robust Face Recognition Based on Mask Learning With Pairwise Differential Siamese Network, ICCV, 2021. </br> 
[[Fabbri et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Fabbri_MOTSynth_How_Can_Synthetic_Data_Help_Pedestrian_Detection_and_Tracking_ICCV_2021_paper.pdf) MOTSynth: How Can Synthetic Data Help Pedestrian Detection and Tracking?, ICCV, 2021.

#### Pedestrain detection
[[Wang et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf)  Repulsion Loss: Detecting Pedestrians in a Crowd, CVPR, 2018.  </br> 
[[Zhang et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper.pdf)  Occlusion-aware R-CNN: Detecting Pedestrians in a Crowd, ECCV, 2018.  </br> 
[[Liu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.pdf) Adaptive NMS: Refining Pedestrian Detection in a Crowd, CVPR, 2019.  </br> 
[[Zhou et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630766.pdf) Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems, ECCV, 2020. </br> 
[[Wu et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Temporal-Context_Enhanced_Detection_of_Heavily_Occluded_Pedestrians_CVPR_2020_paper.pdf) Temporal-Context Enhanced Detection of Heavily Occluded Pedestrians, CVPR, 2020.   </br> 
[[Wu  et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_Where_What_Whether_Multi-Modal_Learning_Meets_Pedestrian_Detection_CVPR_2020_paper.pdf) Where, What, Whether: Multi-modal Learning Meets Pedestrian Detection, CVPR, 2020.  </br> 
[[Huang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_NMS_by_Representative_Region_Towards_Crowded_Pedestrian_Detection_by_Proposal_CVPR_2020_paper.pdf) NMS by Representative Region: Towards Crowded Pedestrian Detection by Proposal Pairing, CVPR, 2020.  </br> 
[[Wang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_Human-Object_Interaction_Detection_Using_Interaction_Points_CVPR_2020_paper.pdf) Learning Human-Object Interaction Detection using Interaction Points, CVPR, 2020.  </br> 
[[Sundararaman et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Sundararaman_Tracking_Pedestrian_Heads_in_Dense_Crowd_CVPR_2021_paper.pdf) Tracking Pedestrian Heads in Dense Crowd, CVPR, 2020.  </br> 
[[Yan et al. 20]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Anchor-Free_Person_Search_CVPR_2021_paper.pdf)  Anchor-Free Person Search, CVPR,2020.  </br> 
[[Gu et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Jiayuan_Gu_Learning_Region_Features_ECCV_2018_paper.pdf) Learning Region Features for Object Detection, ECCV, 2018.

### Image Segmentation

[[Long et al. 15]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) Fully Convolutional Networks for Semantic Segmentation, CVPR, 2015. </br>
[[Noh et al. 15]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf)  Learning Deconvolution Network for Semantic Segmentation, ICCV, 2015.  </br>
[[Badrinarayanan et al. 16]](https://arxiv.org/pdf/1511.00561.pdf)  SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation, ICCV, 2016. </br>
[[Sun et al. 19]](https://arxiv.org/pdf/1904.04514.pdf) High-Resolution Representations for Labeling Pixels and Regions, CVPR, 2019.  </br>
[[Zhao et al. 17]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf) Pyramid Scene Parsing Network, CVPR, 2017. </br>
[[Chen et al. 18]](https://arxiv.org/pdf/1706.05587.pdf) Rethinking Atrous Convolution for Semantic Image Segmentation (Deeplabv3), CVPR, 2018. </br>
[[Visin et al. 16]](https://arxiv.org/pdf/1511.07053v3.pdf) ReSeg: A Recurrent Neural Network-based Model for Semantic Segmentation, CVPR, 2016.  </br>
[[Visin et al. 15]](url) ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks, NIPS, 2015. </br>
[[Chen et al. 16]](https://arxiv.org/pdf/1511.03339.pdf), Attention to Scale: Scale-aware Semantic Image Segmentation, CVPR, 2016.
[[Ghiasi et al. 16]](https://arxiv.org/pdf/1605.02264.pdf) Laplacian Pyramid Reconstruction and Refinement for Semantic Segmentation, ECCV, 2016.
[[Li et al. 18]](https://arxiv.org/pdf/1805.10180.pdf) Pyramid Attention Network for Semantic Segmentation, BMVC, 2018. </br>
[[Fu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf)  Dual Attention Network for Scene Segmentation, CVPR, 2019. </br>
[[Chen et al. 16]](https://arxiv.org/pdf/1511.03339.pdf)  Attention to Scale: Scale-aware Semantic Image Segmentation, ICCV, 2016. </br>
[[Wang et al. 20]](https://arxiv.org/pdf/1908.07919.pdf)  Deep High-Resolution Representation Learning for Visual Recognition, CVPR, 2020. </br>
[[He et al. 17]](https://arxiv.org/pdf/1703.06870.pdf) Mask R-CNN, ICCV, 2017. </br>
[[Yuan et al. 18]](https://arxiv.org/pdf/1809.00916v1.pdf)  OCNet: Object Context Network for Scene Parsing, CVPR, 2019. </br>
[[Wang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Dual_Super-Resolution_Learning_for_Semantic_Segmentation_CVPR_2020_paper.pdf) Dual Super-Resolution Learning for Semantic Segmentation, CVPR, 2020. </br>
[[Liu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.pdf) Structured Knowledge Distillation for Semantic Segmentation, CVPR, 2019. </br>
[[Wang et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520341.pdf)  Intra-class Feature Variation Distillation for Semantic Segmentation, ECCV, 2020. </br>
[[Xu et al. 18]](https://arxiv.org/pdf/2003.06849.pdf)  Deep Affinity Net: Instance Segmentation via Affinity,ECCV, 2018. </br>
[[Quali et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ouali_Semi-Supervised_Semantic_Segmentation_With_Cross-Consistency_Training_CVPR_2020_paper.pdf) Semi-Supervised Semantic Segmentation with Cross-Consistency Training, CVPR, 2020. </br>
[[Zhao et al. 19]](https://proceedings.neurips.cc/paper/2019/file/db9ad56c71619aeed9723314d1456037-Paper.pdf) Multi-source Domain Adaptation for Semantic Segmentation, NIPS, 2019. </br>
[[Chen et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_CrDoCo_Pixel-Level_Domain_Transfer_With_Cross-Domain_Consistency_CVPR_2019_paper.pdf) CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency, CVPR, 2019. </br>
[[Choi et al. 19]](https://arxiv.org/pdf/1909.00589.pdf) Self-Ensembling with GAN-based Data Augmentation for Domain Adaptation in Semantic Segmentation, ICCV, 2019. </br>
[[Xu et al. 19]](https://ojs.aaai.org//index.php/AAAI/article/view/4500) Self-Ensembling Attention Networks: Addressing Domain Shift for Semantic Segmentation, AAAI, 2019. </br>
[[Csurka et al. 21]](https://arxiv.org/pdf/2112.03241.pdf) Unsupervised Domain Adaptation for Semantic Image Segmentation: a Comprehensive Survey, Arxiv, 2021.  </br>
[[Araslanov et al. 21]](https://arxiv.org/pdf/2105.00097.pdf) Self-supervised Augmentation Consistency for Adapting Semantic Segmentation, CVPR, 2021. </br>
[[Chan et al. 20]](https://arxiv.org/pdf/1912.11186.pdf) A Comprehensive Analysis of Weakly-Supervised Semantic Segmentation in Different Image Domains, IJCV, 2020. </br>
[[He et al. 21]](https://arxiv.org/pdf/2103.05423.pdf) Deep Learning based 3D Segmentation: A Survey, Arxiv, 2021. </br>
[[Minaee et al. 20]](https://arxiv.org/pdf/2001.05566v1.pdf) Image Segmentation Using Deep Learning: A Survey, Arxiv, 2020.  </br>

-------Student's reading list----------------------------
 
[[Huang et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_CCNet_Criss-Cross_Attention_for_Semantic_Segmentation_ICCV_2019_paper.pdf) CCNet: Criss-Cross Attention for Semantic Segmentation, ICCV, 2019. </br>
[[Zhu et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhu_Asymmetric_Non-Local_Neural_Networks_for_Semantic_Segmentation_ICCV_2019_paper.pdf) Asymmetric Non-local Neural Networks for Semantic Segmentation, ICCV, 2019. </br>
[[Du et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Du_SSF-DAN_Separated_Semantic_Feature_Based_Domain_Adaptation_Network_for_Semantic_ICCV_2019_paper.pdf) SSF-DAN: Separated Semantic Feature based Domain Adaptation Network for Semantic Segmentation, ICCV, 2019. </br>
[[Ibrahim et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ibrahim_Semi-Supervised_Semantic_Image_Segmentation_With_Self-Correcting_Networks_CVPR_2020_paper.pdf) Semi-Supervised Semantic Image Segmentation with Self-correcting Networks, CVPR,2020.  </br>
[[He et al. 21]](https://arxiv.org/pdf/2103.04717.pdf) Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation, CVPR, 2021. </br>
[[Liu et al. 21]](https://arxiv.org/pdf/2103.16372.pdf)  Source-Free Domain Adaptation for Semantic Segmentation, CVPR, 2021. </br>
[[Liu et al. 21]](https://arxiv.org/pdf/2104.11056.pdf) Domain Adaptation for Semantic Segmentation via Patch-Wise Contrastive Learning, ICCV, 2021. </br>
[[Chen et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_ROAD_Reality_Oriented_CVPR_2018_paper.pdf) ROAD: Reality Oriented Adaptation for Semantic Segmentation of Urban Scenes, CVPR, 2018. </br>
[[Wang et al. 21]](https://arxiv.org/pdf/2009.08610.pdf) Consistency Regularization with High-dimensional Non-adversarial Source-guided Perturbation for Unsupervised Domain Adaptation in Segmentation, AAAI, 2021. </br>
[[Kundu et al. 21]](https://arxiv.org/pdf/2108.11249.pdf) Generalize then Adapt: Source-Free Domain Adaptive Semantic Segmentation, ICCV, 2021. </br>
[[Wang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.pdf)
Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation, CVPR, 2020. </br>
[[Sun et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_ECS-Net_Improving_Weakly_Supervised_Semantic_Segmentation_by_Using_Connections_Between_ICCV_2021_paper.pdf) ECS-Net: Improving Weakly Supervised Semantic Segmentation by Using Connections Between Class Activation Maps, ICCV, 2021. </br>
[[Chang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chang_Weakly-Supervised_Semantic_Segmentation_via_Sub-Category_Exploration_CVPR_2020_paper.pdf) Weakly-Supervised Semantic Segmentation via Sub-category Exploration, CVPR, 2020. </br>
[[He et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/He_GANSeg_Learning_To_Segment_by_Unsupervised_Hierarchical_Image_Generation_CVPR_2022_paper.pdf) GANSeg: Learning to Segment by Unsupervised Hierarchical Image Generation, CVPR, 2022.</br>
[[Zhou et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Rethinking_Semantic_Segmentation_A_Prototype_View_CVPR_2022_paper.pdf) Rethinking Semantic Segmentation: A Prototype View, CVPR, 2022. </br>
[[Peng et al. 22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.pdf) Semantic-Aware Domain Generalized Segmentation, CVPR, 2022.  </br>
[[Zhou et al. 22]](https://arxiv.org/pdf/2205.02833.pdf) Cross-view Transformers for real-time Map-view Semantic Segmentation, CVPR, 2022. </br>
[[Wu et al. 22]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890443.pdf) Dynamic Density-aware Active Domain Adaptation, ECCV, 2022. </br>

### Computer vision with novel camera sensors  (1)- Event-based vision
[[Zhang et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630647.pdf)  Learning to See in the Dark with Events, ECCV, 2020.  </br>
[[Rebacq et al. 19]](https://rpg.ifi.uzh.ch/docs/TPAMI19_Rebecq.pdf)  High Speed and High Dynamic Range Video with an Event Camera, IEEE TPAMI (CVPR), 2019.  </br>
[[Wang et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580154.pdf) Event Enhanced High-Quality Image Recovery, ECCV, 2020.  </br>
[[Wang et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Event-Based_High_Dynamic_Range_Image_and_Very_High_Frame_Rate_CVPR_2019_paper.pdf) Event-based High Dynamic Range Image and Very High Frame Rate Video Generation using Conditional Generative Adversarial Networks, CVPR, 2019.  </br>
[[Wang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_EventSR_From_Asynchronous_Events_to_Image_Reconstruction_Restoration_and_Super-Resolution_CVPR_2020_paper.pdf) EventSR: From Asynchronous Events to Image Reconstruction, Restoration, and Super-Resolution via End-to-End Adversarial Learning, CVPR, 2020. </br> 
[[Kim et al. 22]](https://arxiv.org/pdf/2112.06988v2.pdf)  Event-guided Deblurring of Unknown Exposure Time Videos, Arxiv, 2022. </br>
[[Mostafavi et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf) Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds, ICCV, 2021.  </br>
[[Wang et al. 21]](https://arxiv.org/pdf/2109.01801.pdf)  Dual Transfer Learning for Event-based End-task Prediction via Pluggable Event to Image Translation, ICCV, 2021. </br>
[[Han et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Han_EvIntSR-Net_Event_Guided_Multiple_Latent_Frames_Reconstruction_and_Super-Resolution_ICCV_2021_paper.pdf) EvIntSR-Net: Event Guided Multiple Latent Frames Reconstruction and Super-resolution, ICCV, 2021. </br> 
[[Gehrig et al. 21]](https://arxiv.org/pdf/2102.09320.pdf) Combining Events and Frames using Recurrent Asynchronous Multimodal Networks for Monocular Depth Prediction, ICRA, 2021. </br> 
[[Alonso et al. 19]](https://openaccess.thecvf.com/content_CVPRW_2019/papers/EventVision/Alonso_EV-SegNet_Semantic_Segmentation_for_Event-Based_Cameras_CVPRW_2019_paper.pdf) EV-SegNet: Semantic Segmentation for Event-based Cameras, CVPR, 2019. </br> 
[[Xu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Motion_Deblurring_With_Real_Events_ICCV_2021_paper.pdf) Motion Deblurring with Real Events, ICCV, 2021. </br> 

---
[[Lin et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530681.pdf)  Learning Event-Driven Video Deblurring and Interpolation, ECCV, 2020. </br>
[[Federico et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Paredes-Valles_Back_to_Event_Basics_Self-Supervised_Learning_of_Image_Reconstruction_for_CVPR_2021_paper.pdf)  Back to Event Basics: Self-Supervised Learning of Image Reconstruction for Event Cameras via Photometric Constancy, CVPR, 2021. </br>
[[Jing et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Jing_Turning_Frequency_to_Resolution_Video_Super-Resolution_via_Event_Cameras_CVPR_2021_paper.pdf)  Turning Frequency to Resolution: Video Super-resolution via Event Cameras, CVPR, 2021. </br>
[[Zou et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zou_Learning_To_Reconstruct_High_Speed_and_High_Dynamic_Range_Videos_CVPR_2021_paper.pdf) Learning to Reconstruct High Speed and High Dynamic Range Videos from Events, CVPR, 2021. </br>
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Indoor_Lighting_Estimation_Using_an_Event_Camera_CVPR_2021_paper.pdf)  Indoor Lighting Estimation using an Event Camera, CVPR, 2021. </br>
[[Zhang et al. 21]](https://arxiv.org/pdf/2103.02376.pdf) Event-based Synthetic Aperture Imaging with a Hybrid Network, CVPR, 2021. </br>
[[Tulyakov et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Tulyakov_Time_Lens_Event-Based_Video_Frame_Interpolation_CVPR_2021_paper.pdf)  Time Lens: Event-based Video Frame Interpolation, CVPR, 2021. </br>
[[Shang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Shang_Bringing_Events_Into_Video_Deblurring_With_Non-Consecutively_Blurry_Frames_ICCV_2021_paper.pdf) Bringing Events into Video Deblurring with Non-consecutively Blurry Frames, ICCV, 2021. </br>
[[Xu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Motion_Deblurring_With_Real_Events_ICCV_2021_paper.pdf) Motion Deblurring with Real Events, ICCV, 2021. </br>
[[Yu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Training_Weakly_Supervised_Video_Frame_Interpolation_With_Events_ICCV_2021_paper.pdf)  Training Weakly Supervised Video Frame Interpolation with Events, ICCV, 2021.  </br>
[[Li et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Event_Stream_Super-Resolution_via_Spatiotemporal_Constraint_Learning_ICCV_2021_paper.pdf)  Event Stream Super-Resolution via Spatiotemporal Constraint Learning, ICCV, 2021. </br>
 [[Weng et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Weng_Event-Based_Video_Reconstruction_Using_Transformer_ICCV_2021_paper.pdf) Event-based Video Reconstruction Using Transformer, ICCV, 2021. </br>
 [[Zou et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zou_EventHPE_Event-Based_3D_Human_Pose_and_Shape_Estimation_ICCV_2021_paper.pdf) EventHPE: Event-based 3D Human Pose and Shape Estimation, ICCV, 2021. </br>
[[Zhang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Object_Tracking_by_Jointly_Exploiting_Frame_and_Event_Domain_ICCV_2021_paper.pdf) Object Tracking by Jointly Exploiting Frame and Event Domain, ICCV, 2021.  </br>
 
 
 ### Depth and Motion Estimation in Vision 
#### Depth Estimation (Lecture notes)
[[Ming et al. 21]](https://www-sciencedirect-com.lib.ezproxy.ust.hk/science/article/pii/S0925231220320014) Deep learning for monocular depth estimation: A review, Neurocomputing, 2021.  </br>
[[Eigen et al.]](https://arxiv.org/pdf/1406.2283.pdf), “Depth Map Prediction from a Single Image using a Multi-Scale Deep Network”, NeurIPS, 2014. </br>
[[Laina et al. 16]](https://arxiv.org/pdf/1606.00373.pdf) Deeper depth prediction with fully convolutional residual networks, 3D vision,2016.  </br> 
[[Fu et al. 18]](https://arxiv.org/pdf/1806.02446.pdf) Deep Ordinal Regression Network for Monocular Depth Estimation, CVPR, 2018. </br>
[[Ren et al. 18]](https://arxiv.org/pdf/1803.08669.pdf) Pyramid Stereo Matching Network, CVPR, 2018.  </br>
[[Jung et al. 17]](url) Depth prediction from a single image with conditional adversarial networks, ICIP, 2017. </br>

#### Motion Estimation (Optical Flow) (Lecture notes)
[[Dosovitskiy et al. 15]](https://openaccess.thecvf.com/content_iccv_2015/papers/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.pdf) Flownet: Learning optical flow with convolutional networks, ICCV, 2015. </br>
[[Ilg et al. 15]](https://arxiv.org/pdf/1612.01925.pdf) FlowNet 2.0: Evolution of Optical Flow Estimation With Deep Networks, CVPR, 2017. </br>
[[[Ilg et al. 18]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Eddy_Ilg_Occlusions_Motion_and_ECCV_2018_paper.pdf) Occlusions, Motion and Depth Boundaries with a Generic Network for Disparity, Optical Flow or Scene Flow Estimation, ECCV, 2018. </br>
[[Ranjan et al. 17]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ranjan_Optical_Flow_Estimation_CVPR_2017_paper.pdf) Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017. </br>

#### SLAM (Lecture notes)
[[Raul et al. 16]](https://arxiv.org/abs/1610.06475) ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras, IEEE Transactions on Robotics, 2016. </br>

#### --------------------------------------Depth and Motion  Estimation   (Students' reading list)--------------------------------------------------------------
[[Xu et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Structured_Attention_Guided_CVPR_2018_paper.pdf) Structured Attention Guided Convolutional Neural Fields for Monocular Depth Estimation, CVPR, 2018. </br>
[[Godard et al. 17]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.pdf?msclkid=4e0b72c4b31311ec8379d9fffe94ff53)  Unsupervised Monocular Depth Estimation with Left-Right Consistency, CVPR, 2017. </br>
[[Kuznietsov et al. 17]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Kuznietsov_Semi-Supervised_Deep_Learning_CVPR_2017_paper.pdf?msclkid=ab7379b1b31311ec9829857fbccfca4f) Semi-Supervised Deep Learning for Monocular Depth Map Prediction, CVPR, 2017.  </br>
[[Pilzer et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pilzer_Refine_and_Distill_Exploiting_Cycle-Inconsistency_and_Knowledge_Distillation_for_Unsupervised_CVPR_2019_paper.pdf?msclkid=d7854ed1b31311ec9828b9837799361e)  Refine and Distill: Exploiting Cycle-Inconsistency and Knowledge Distillation for Unsupervised Monocular Depth Estimation, CVPR, 2019. </br>
[[Cun et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580732.pdf?msclkid=d7851131b31311eca4b25ebb634ad62e)  Defocus Blur Detection via Depth Distillation, ECCV, 2020. </br>
[[Ranftl et al. 21]](https://arxiv.org/pdf/2103.13413v1.pdf) Vision Transformers for Dense Prediction, CVPR, 2021. </br>
[[Meng et al. 19]](https://arxiv.org/pdf/1812.05642v2.pdf) SIGNet: Semantic Instance Aided Unsupervised 3D Geometry Perception, CVPR, 2019. </br>
[[Liu et al. 21]](https://arxiv.org/pdf/2108.07628v1.pdf)  Self-supervised Monocular Depth Estimation for All Day Images using Domain Separation, ICCV, 2021. </br>
[[Huynh et al. 20]](https://arxiv.org/pdf/2004.02760v2.pdf)  Guiding Monocular Depth Estimation Using Depth-Attention Volume, ECCV, 2020. </br>
[[Watson et al. 20]](https://arxiv.org/pdf/2008.01484v2.pdf) Learning Stereo from Single Images, ECCV, 2020. </br>
[[YUan et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Efficient_Dynamic_Scene_Deblurring_Using_Spatially_Variant_Deconvolution_Network_With_CVPR_2020_paper.pdf?msclkid=405bb458b31611ecbbade73ec6204312), Efficient Dynamic Scene Deblurring Using Spatially Variant Deconvolution
Network with Optical Flow Guided Training, CVPR, 2020. </br>
[[Yan et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yan_Optical_Flow_in_Dense_Foggy_Scenes_Using_Semi-Supervised_Learning_CVPR_2020_paper.pdf?msclkid=8d28ca4cb31611ec8afd797ae1ef388e) Optical Flow in Dense Foggy Scenes Using Semi-Supervised Learning, CVPR, 2020.  </br>
[[Aleotti et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Aleotti_Learning_Optical_Flow_From_Still_Images_CVPR_2021_paper.pdf?msclkid=c3116ce6b31611ecb7331cd1de5688a6)  Learning optical flow from still images, CVPR, 2021. </br>
[[Luo et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_UPFlow_Upsampling_Pyramid_for_Unsupervised_Optical_Flow_Learning_CVPR_2021_paper.pdf?msclkid=c3118a04b31611ec99b8b3dd7bca20c2)  UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning, CVPR, 2021. </br>

### NeRF + SLAM ----
[[Lin et al. 22]](https://arxiv.org/pdf/2012.05877.pdf)  iNeRF Inverting Neural Radiance Fields for Pose Estimation, IROS, 2021. </br>
[[Sucar et al. 21]](https://arxiv.org/abs/2103.12352)  iMAP: licit Mapping and Positioning in Real-Time, ICCV, 2021. </br>
[[Lin et al. 22]](https://arxiv.org/abs/2104.06405)  BARF : Bundle-Adjusting Neural Radiance Fields, ICCV, 2021. </br>
[[Luo et al. 21]](https://arxiv.org/pdf/2210.13641.pdf)  NeRF-SLAM: Real-Time Dense Monocular SLAM with Neural Radiance Fields, CVPR, 2021. </br>
[[Zhu et al. 22]](https://arxiv.org/abs/2112.12130)  NICE-SLAM: Neural Implicit Scalable Encoding for SLAM, CVPR, 2022. </br>
[[Li et al. 23]](https://openreview.net/forum?id=QUK1ExlbbA)  Neural Implicit Scalable Encoding for SLAM, ICLR, 2023 (first RGB-based approach). </br>
[[Johari et al. 23]](https://openreview.net/forum?id=QUK1ExlbbA)  ESLAM: Efficient Dense SLAM System Based on Hybrid Representation of Signed Distance Fields, CVPR 2023 Highlight (coarse-to-fine approach). </br>

### Computer vision with novel camera sensors (II)
[[Kuang et al. 19]](https://arxiv.org/ftp/arxiv/papers/1810/1810.05399.pdf) Thermal Infrared Colorization via Conditional Generative Adversarial Network， ICCP, 2019. </br>
[[Nniaz et al.20]](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Kniaz_ThermalGAN_Multimodal_Color-to-Thermal_Image_Translation_for_Person_Re-Identification_in_Multispectral_ECCVW_2018_paper.pdf) ThermalGAN: Multimodal Color-to-Thermal Image Translation for Person Re-Identification in Multispectral Dataset, ECCV,2018.</br> 
[[Li et al. 19]](https://arxiv.org/pdf/1907.10303.pdf) Segmenting Objects in Day and Night: Edge-Conditioned CNN for Thermal Image Semantic Segmentation, IEEE TNNLS, 2019. </br> 
[[Wang et al. 20]](https://arxiv.org/pdf/2002.04114.pdf) Cross-Modality Paired-Images Generation for RGB-Infrared Person Re-Identification, AAAI,2020. </br> 
[[Deng et al. 21]](https://arxiv.org/pdf/2110.08988.pdf) FEANet: Feature-Enhanced Attention Network for RGB-Thermal Real-time Semantic Segmentation, ICRA,  2021.  </br> 
[[Sun et al. 20]](https://hlwang1124.github.io/data/sun2020fuseseg.pdf) FuseSeg: Semantic Segmentation of Urban Scenes Based on RGB and Thermal Data Fusion, IEEE TASE, 2020. </br> 
[[Zhang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_ABMDRNet_Adaptive-Weighted_Bi-Directional_Modality_Difference_Reduction_Network_for_RGB-T_Semantic_CVPR_2021_paper.pdf) ABMDRNet: Adaptive-weighted Bi-directional Modality Difference Reduction Network for RGB-T Semantic Segmentation, CVPR, 2021. </br> 

#### 360 vision
[[Wang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_BiFuse_Monocular_360_Depth_Estimation_via_Bi-Projection_Fusion_CVPR_2020_paper.pdf) BiFuse: Monocular 360◦ Depth Estimation via Bi-Projection Fusion, CVPR, 2020. </br> 
[[Deng et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_LAU-Net_Latitude_Adaptive_Upscaling_Network_for_Omnidirectional_Image_Super-Resolution_CVPR_2021_paper.pdf) LAU-Net: Latitude Adaptive Upscaling Network for Omnidirectional Image Super-resolution, CVPR, 2021. </br> 
[[Lee et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_SpherePHD_Applying_CNNs_on_a_Spherical_PolyHeDron_Representation_of_360deg_CVPR_2019_paper.pdf) SpherePHD: Applying CNNs on a Spherical PolyHeDron Representation of 360◦ Images, CVPR, 2019.  </br> 
[[Cohen et al. 18]](https://arxiv.org/pdf/1801.10130.pdf) SPHERICAL CNNS, ICLR, 2018. </br>
[[Chen et al. 18]](https://arxiv.org/pdf/1806.01320.pdf) Cube Padding for Weakly-Supervised Saliency Prediction in 360◦ Videos, CVPR, 2018.  </br>
[[Jeon et al. 18]](http://cg.postech.ac.kr/papers/2018_ACCV_Jeon.pdf)  Deep Upright Adjustment of 360 Panoramas Using Multiple Roll Estimations, ACCV, 2018 </br>
[[Davidson et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730579.pdf) 360o Camera Alignment via Segmentation, ECCV, 2020. </br>
[[Su et al. 18]](https://proceedings.neurips.cc/paper/2017/file/0c74b7f78409a4022a2c4c5a5ca3ee19-Paper.pdf) Learning Spherical Convolution for Fast Features from 360° Imagery, NIPS, 2018. </br>
[[Tateno et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Keisuke_Tateno_Distortion-Aware_Convolutional_Filters_ECCV_2018_paper.pdf) Distortion-Aware Convolutional Filters for Dense Prediction in Panoramic Images, ECCV, 2018.  </br>

---- 
#### Thermal camera-based vision (reading list)

[[Ghose et al. 19]](https://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Ghose_Pedestrian_Detection_in_Thermal_Images_Using_Saliency_Maps_CVPRW_2019_paper.pdf) Pedestrian Detection in Thermal Images using Saliency Maps, CVPR, 2019. </br> 
[[Kieu et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf) Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery, ECCV, 2020. </br> 
[[Li et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700460.pdf) Full-Time Monocular Road Detection Using Zero-Distribution Prior of Angle of Polarization, ECCV, 2020. </br> 
[[Choi et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Choi_Hi-CMD_Hierarchical_Cross-Modality_Disentanglement_for_Visible-Infrared_Person_Re-Identification_CVPR_2020_paper.pdf) Hi-CMD: Hierarchical Cross-Modality Disentanglement for Visible-Infrared Person Re-Identification, CVPR, 2020.</br> 
[[Wu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Discover_Cross-Modality_Nuances_for_Visible-Infrared_Person_Re-Identification_CVPR_2021_paper.pdf)  Discover Cross-Modality Nuances for Visible-Infrared Person Re-Identification, CVPR, 2021. </br>  
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Neural_Feature_Search_for_RGB-Infrared_Person_Re-Identification_CVPR_2021_paper.pdf)  Neural Feature Search for RGB-Infrared Person Re-Identification, CVPR, 2021. </br>  
[[Ye et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Ye_Channel_Augmented_Joint_Learning_for_Visible-Infrared_Recognition_ICCV_2021_paper.pdf) Channel Augmented Joint Learning for Visible-Infrared Recognition, ICCV, 2021. </br> 
[[Fu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Fu_CM-NAS_Cross-Modality_Neural_Architecture_Search_for_Visible-Infrared_Person_Re-Identification_ICCV_2021_paper.pdf) CM-NAS: Cross-Modality Neural Architecture Search for Visible-Infrared Person Re-Identification, ICCV, 2021. </br>  
[[Wei et al.21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wei_Syncretic_Modality_Collaborative_Learning_for_Visible_Infrared_Person_Re-Identification_ICCV_2021_paper.pdf) Syncretic Modality Collaborative Learning for Visible Infrared Person Re-Identification, ICCV, 2021. </br>  
[[Park et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Learning_by_Aligning_Visible-Infrared_Person_Re-Identification_Using_Cross-Modal_Correspondences_ICCV_2021_paper.pdf)  Visible-Infrared Person Re-identification using Cross-Modal Correspondences, ICCV, 2021. </br>  
[[Ye et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620222.pdf) Dynamic Dual-Attentive Aggregation Learning for Visible-Infrared Person Re-Identification, ECCV, 2020. </br>  
[[Kieu et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf)  Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery, ECCV, 2020. </br> 
[[Wu et al. 20]](https://ojs.aaai.org/index.php/AAAI/article/view/5891) Infrared-Visible Cross-Modal Person Re-Identification with an X Modality, AAAI, 2020. </br>  
[[Wang et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_RGB-Infrared_Cross-Modality_Person_Re-Identification_via_Joint_Pixel_and_Feature_Alignment_ICCV_2019_paper.pdf) RGB-Infrared Cross-Modality Person Re-Identification via Joint Pixel and Feature Alignment, ICCV, 2019.</br>  
[[Feng et al. 20]](https://ieeexplore.ieee.org/abstract/document/8765608)  Learning Modality-Specific Representations for Visible-Infrared Person Re-Identification, IEEE TIP, 2020. </br>
[[Ye et al. 20]](https://www.comp.hkbu.edu.hk/~mangye/files/TIP_MACE.pdf) Cross-Modality Person Re-Identification via Modality-aware Collaborative Ensemble Learning, IEEE TIP, 2020. </br>
[[Wang et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_to_Reduce_Dual-Level_Discrepancy_for_Infrared-Visible_Person_Re-Identification_CVPR_2019_paper.pdf) Learning to Reduce Dual-level Discrepancy for Infrared-Visible Person Re-identification, CVPR, 2019. </br>
  
#### 360 vision (reading list)
[[Jin et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Geometric_Structure_Based_and_Regularized_Depth_Estimation_From_360_Indoor_CVPR_2020_paper.pdf) Geometric Structure Based and Regularized Depth Estimation From 360◦ Indoor Imagery, CVPR, 2020.  </br> 
[[Deng et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_LAU-Net_Latitude_Adaptive_Upscaling_Network_for_Omnidirectional_Image_Super-Resolution_CVPR_2021_paper.pdf) LAU-Net: Latitude Adaptive Upscaling Network for Omnidirectional Image Super-resolution, CVPR, 2021. </br>
[[Sun et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_HoHoNet_360_Indoor_Holistic_Understanding_With_Latent_Horizontal_Features_CVPR_2021_paper.pdf) HoHoNet: 360 Indoor Holistic Understanding with Latent Horizontal Features, CVPR, 2021. </br>
[[Yang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Capturing_Omni-Range_Context_for_Omnidirectional_Segmentation_CVPR_2021_paper.pdf) Capturing Omni-Range Context for Omnidirectional Segmentation, CVPR, 2021. </br>
[[Yang et al. 21]](https://ieeexplore.ieee.org/abstract/document/9321183?casa_token=HmK2DmzfMqYAAAAA:aNRR_akT-ex4DGE1uzKl91K9ucSXNYqMxcWWE9iwqnr5iR1RpCpakCoS5tQyaHDKFCROLG4) Is Context-Aware CNN Ready for the Surroundings? Panoramic Semantic Segmentation in the Wild, IEEE TIP, 2021. </br>
[[Li et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Looking_Here_or_There_Gaze_Following_in_360-Degree_Images_ICCV_2021_paper.pdf) Looking here or there? Gaze Following in 360-Degree Images, ICCV, 2021.  </br>
[[Djilali et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Djilali_Rethinking_360deg_Image_Visual_Attention_Modelling_With_Unsupervised_Learning._ICCV_2021_paper.pdf) Rethinking 360° Image Visual Attention Modelling with Unsupervised Learning, ICCV, 2021. </br>
[[Tran et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Tran_SSLayout360_Semi-Supervised_Indoor_Layout_Estimation_From_360deg_Panorama_CVPR_2021_paper.pdf) SSLayout360: Semi-Supervised Indoor Layout Estimation from 360◦ Panorama, CVPR, 2021. </br>



### Adversarial Robustness in Computer Vision 
[[Goodfellow et al. 15]](https://arxiv.org/pdf/1412.6572.pdf) Explaining and harnessing adversarial examples, ICLR, 2015.  </br>
[[Szegedy et al. 14]](https://arxiv.org/pdf/1312.6199.pdf) Intriguing properties of neural networks, ICLR, 2014. </br>
[[Su et al. 17]](https://arxiv.org/pdf/1710.08864.pdf) One pixel attack for fooling deep neural networks, Arxiv, 2017. </br>
[[Karmon et al. 18]](http://proceedings.mlr.press/v80/karmon18a/karmon18a.pdf) LaVAN: Localized and Visible Adversarial Noise, ICML, 2018. </br>
[[Xie et al. 17]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Xie_Adversarial_Examples_for_ICCV_2017_paper.pdf) Adversarial Examples for Semantic Segmentation and Object Detection, ICCV, 2017. </br>
[[Moosavi et al. 17]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf) Universal adversarial perturbations, ICCV, 2017. </br>
[[Poursaeed et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.pdf)  Generative Adversarial Perturbations, CVPR, 2018. </br>
[[Chen et al. 18]](https://arxiv.org/pdf/1804.05810.pdf)  ShapeShifter: Robust Physical Adversarial Attack on Faster R-CNN Object Detector, ECML PKDD, 2018. </br>
[[Chao et al. 19]](https://openreview.net/pdf?id=HknbyQbC-) Generating Adversarial Examples with Adversarial Networks, IJCAI, 2019. </br>
[[Wang et al. 21]](https://ieeexplore.ieee.org/document/9524508) Psat-gan: Efficient adversarial attacks against holistic scene understanding, IEEE TIP, 2021. </br>
[[Carli et al. 17]](https://arxiv.org/pdf/1608.04644.pdf)  Towards Evaluating the Robustness of Neural Networks, Axiv, 2017. </br>
[[Xiao et al. 18]](https://arxiv.org/pdf/1801.02612.pdf)  SPATIALLY TRANSFORMED ADVERSARIAL EXAMPLES, ICLR,2018. </br>

#### Reading list 
---------------------------
[[Zhou et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_DaST_Data-Free_Substitute_Training_for_Adversarial_Attacks_CVPR_2020_paper.pdf) DaST: Data-Free Substitute Training for Adversarial Attacks, CVPR, 2020. </br>
[[Naseer et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Naseer_A_Self-supervised_Approach_for_Adversarial_Robustness_CVPR_2020_paper.pdf) A Self-supervised Approach for Adversarial Robustness, CVPR, 2020. </br>
[[Zi et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zi_Revisiting_Adversarial_Robustness_Distillation_Robust_Soft_Labels_Make_Student_Better_ICCV_2021_paper.pdf) 
Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better, CVPR, 2021. </br>
[[Mahmood et al. 21]](https://arxiv.org/pdf/2104.02610v2.pdf) On the Robustness of Vision Transformers to Adversarial Examples, ICCV, 2021.  </br>
[[Wang et al. 21]](https://arxiv.org/pdf/2107.14185.pdf) Feature Importance-aware Transferable Adversarial Attacks, ICCV, 2021.  </br>
[[Mao et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470154.pdf) Multitask Learning Strengthens Adversarial Robustness, ECCV, 2020 </br>
[[Arnab et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Arnab_On_the_Robustness_CVPR_2018_paper.pdf) On the Robustness of Semantic Segmentation Models to Adversarial Attacks, CVPR, 2018.  </br>
[[He et al. 19]](https://arxiv.org/pdf/1904.12181.pdf) Biomedical Image Segmentation against Adversarial Attacks, AAAI, 2019.  </br>
[[Joshi et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Joshi_Semantic_Adversarial_Attacks_Parametric_Transformations_That_Fool_Deep_Classifiers_ICCV_2019_paper.pdf) Semantic Adversarial Attacks: Parametric Transformations That Fool Deep Classifiers, CVPR, 2019. </br>
[[Shamsabadi et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shamsabadi_ColorFool_Semantic_Adversarial_Colorization_CVPR_2020_paper.pdf)  ColorFool: Semantic Adversarial Colorization, CVPR, 2020.

<!-- 
### Scene Understanding in adverse vision conditions 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_HLA-Face_Joint_High-Low_Adaptation_for_Low_Light_Face_Detection_CVPR_2021_paper.pdf)  HLA-Face: Joint High-Low Adaptation for Low Light Face Detection, CVPR, 2021.
[[Lin et al. 20]](https://arxiv.org/pdf/1909.03403.pdf) Open Compound Domain Adaptation, CVPR, 2020.
 -->



