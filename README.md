# DeepDomers
MTMA 2016 project using Tensor Flow

Slides: https://docs.google.com/presentation/d/19nqjC4jGxJAunwMyTuLftIJ8PCfT7lKLMwEb7NqjK5Q/edit?usp=sharing

#End-to-end Morphology-aware Neural MT (Austin Matthews)
Room 333
Current neural MT methods rely on naïve word representations that make no use of available sub-word information. Previous approaches, such as factored translation, have shown that such information can greatly help translation quality, especially between certain language pairs. We will present and implement a novel method of building morphology-aware word embeddings on both the source and target sides, as well as a probabilistically well-formed method to combine word- and subword-level model probabilities for output sequences


#to-do's
- encoder
- decoder
- + attention
- synthetic data? maybe spanish (without gender? + be able to add that back in
- find spanish analyzer DONE
- processing/cleaning/tokenization
  - Spanish-English corpus + Morphology is ready
- What to do about different beams for different types of generation
- evaluation
- morphemes --> surface form (do we have this for spanish?)
