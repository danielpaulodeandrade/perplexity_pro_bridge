# PPRO Bridge Prototype

Bridge local OpenAI-compatible para usar Perplexity Web como backend de chat
em Continue e Trae.

## MVP

- Perplexity Web como único backend
- Playwright com perfil persistente isolado
- API local em loopback
- OpenAI Chat Completions não-streaming
- Sem tools, shell, Git, escrita de arquivos ou execução automática

## Segurança

- Perfil do navegador não é versionado
- Bridge escuta somente em 127.0.0.1
- Respostas do chat são texto não confiável
- Git e execução permanecem sob controle humano
