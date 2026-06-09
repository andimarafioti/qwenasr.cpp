import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from convert import expected_hf_shapes, load_bpe_vocab, should_include_tensor


def _cfg(*, audio_hidden=896, audio_layers=18, text_hidden=1024, text_intermediate=3072):
    return {
        "thinker_config": {
            "audio_config": {
                "num_mel_bins": 128,
                "d_model": audio_hidden,
                "encoder_layers": audio_layers,
                "encoder_attention_heads": audio_hidden // 64,
                "encoder_ffn_dim": audio_hidden * 4,
                "downsample_hidden_size": 480,
                "output_dim": text_hidden,
            },
            "text_config": {
                "vocab_size": 151936,
                "hidden_size": text_hidden,
                "intermediate_size": text_intermediate,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
            },
        }
    }


class ConvertShapeTest(unittest.TestCase):
    def test_expected_shapes_cover_qwen3_asr_0_6b(self):
        shapes = expected_hf_shapes(_cfg())
        self.assertEqual(len(shapes), 612)
        self.assertEqual(shapes["text.blk.0.attn_q.weight"], (2048, 1024))
        self.assertEqual(shapes["text.blk.0.attn_k.weight"], (1024, 1024))
        self.assertEqual(shapes["audio.conv.0.weight"], (480, 1, 3, 3))
        self.assertEqual(shapes["audio.conv_out.weight"], (896, 7680))
        self.assertEqual(shapes["audio.proj.1.weight"], (1024, 896))

    def test_expected_shapes_cover_qwen3_asr_1_7b(self):
        shapes = expected_hf_shapes(
            _cfg(audio_hidden=1024, audio_layers=24, text_hidden=2048, text_intermediate=6144)
        )
        self.assertEqual(len(shapes), 708)
        self.assertEqual(shapes["text.token_embd.weight"], (151936, 2048))
        self.assertEqual(shapes["text.blk.0.ffn_up.weight"], (6144, 2048))
        self.assertEqual(shapes["audio.conv_out.weight"], (1024, 7680))
        self.assertEqual(shapes["audio.proj.1.weight"], (2048, 1024))


class ConvertTokenizerTest(unittest.TestCase):
    def test_load_bpe_vocab_fills_model_vocab_and_added_tokens(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vocab.json").write_text('{"a": 0, "b": 2}\n')
            (root / "merges.txt").write_text("#version: 0.2\na b\n")
            (root / "tokenizer_config.json").write_text(
                '{"added_tokens_decoder": {"4": {"content": "<|im_end|>"}}}\n'
            )

            meta = load_bpe_vocab(root, vocab_size=6)

        self.assertEqual(meta.tokens, ["a", "<|unused-1|>", "b", "<|unused-3|>", "<|im_end|>", "<|unused-5|>"])
        self.assertEqual(meta.token_types, [1, 5, 1, 5, 4, 5])
        self.assertEqual(meta.merges, ["a b"])
        self.assertEqual(meta.token_ids["<|im_end|>"], 4)


class ConvertWriterTest(unittest.TestCase):
    def test_include_prefix_filters_native_tensor_names(self):
        self.assertTrue(should_include_tensor("audio.conv.0.weight", ("audio.conv.0.",)))
        self.assertFalse(should_include_tensor("audio.conv.1.weight", ("audio.conv.0.",)))
        self.assertTrue(should_include_tensor("text.token_embd.weight", ()))


if __name__ == "__main__":
    unittest.main()
