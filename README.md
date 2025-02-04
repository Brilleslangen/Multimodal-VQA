# Multimodal-VGA
In the continuously evolving landscape of artificial intelligence, multi-modality has emerged as the holy grail for pushing the boundaries of what machines are capable of. Both industry and academia are focusing on foundation models that integrate and unify a range of modalities to achieve a more holistic and context-aware form of intelligence. These models represent a significant leap beyond single-domain approaches, enabling a broader understanding of real-world knowledge and unlocking a versatile range of applications.

In this final project of the course *Multimodal Intelligent System Desing* at The University of Tokyo, we explore one of those applications, namely *visual multiple-choice question answering*, constructing and comparing various approaches, built upon the base architecture of the visual language model, **PaliGemma 2**:

 + Conditonal generation with cosine similarity selection
 + Multi-class classification
 + Natural language inference using SWAG-based data decomposition

Additionally, for the latter two approaches, we also assess two distinct pooling strategies for intermediate feature extraction from the final hidden state and the linear output layer of our non-generative approaches - attentive vs. last-token pooling.  
