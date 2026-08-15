// Stream uint16 token shards as document-delimited decimal text for KenLM.
// Token 0 is the tokenizer's end-of-text marker: it remains a scored word at
// the end of its document, then terminates the lmplz input line.

#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: ngram_tokens_to_text SHARD.bin [SHARD.bin ...]\n";
    return 2;
  }
  std::ios::sync_with_stdio(false);
  bool at_line_start = true;
  std::array<std::uint16_t, 1 << 20> buffer{};
  std::array<char, 8> digits{};
  for (int arg = 1; arg < argc; ++arg) {
    std::ifstream input(argv[arg], std::ios::binary);
    if (!input) {
      std::cerr << "cannot open " << argv[arg] << "\n";
      return 1;
    }
    while (input) {
      input.read(reinterpret_cast<char*>(buffer.data()),
                 static_cast<std::streamsize>(buffer.size() * sizeof(std::uint16_t)));
      const auto count = static_cast<std::size_t>(input.gcount()) / sizeof(std::uint16_t);
      for (std::size_t index = 0; index < count; ++index) {
        const std::uint16_t token = buffer[index];
        if (!at_line_start) std::cout.put(' ');
        const auto converted = std::to_chars(digits.data(), digits.data() + digits.size(), token);
        if (converted.ec != std::errc{}) throw std::runtime_error("token conversion failed");
        std::cout.write(digits.data(), converted.ptr - digits.data());
        if (token == 0) {
          std::cout.put('\n');
          at_line_start = true;
        } else {
          at_line_start = false;
        }
      }
      if (input.bad()) {
        std::cerr << "read failure in " << argv[arg] << "\n";
        return 1;
      }
    }
  }
  if (!at_line_start) std::cout.put('\n');
  return 0;
}
