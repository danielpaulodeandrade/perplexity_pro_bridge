# Arquitetura

Continue ou Trae
  -> OpenAI-compatible HTTP
  -> Bridge local
  -> Playwright
  -> Perplexity Web

O MVP aceita apenas chat textual com `stream: false`.

A bridge:
- não executa comandos;
- não lê o workspace automaticamente;
- não altera Git;
- não interpreta tools, YAML, XML ou patches como ações;
- devolve a resposta do Perplexity apenas como texto.
