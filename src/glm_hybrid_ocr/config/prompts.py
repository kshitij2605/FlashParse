CLASSIFICATION_PROMPT = """Classify this image into exactly ONE of these 4 categories:

1. "chart" - Statistical/data visualizations including:
   - Bar graphs, line charts, pie charts, dot charts
   - Area charts, scatter plots, histograms
   - Any graph showing numerical data with axes or legends

2. "figure" - Diagrams and conceptual illustrations including:
   - Flowcharts, process diagrams, organizational charts
   - Architecture diagrams, system diagrams
   - Timelines, workflows, interlinked concept images
   - Infographics explaining a process or relationship

3. "scanned_text" - Text-heavy documents including:
   - Pages containing ONLY printed or written text
   - Documents with simple borders or headers
   - Text blocks without significant visual elements

4. "miscellaneous" - All other images including:
   - Photographs of people, places, animals, objects
   - Company logos, brand images, decorative images
   - Screenshots, product images, promotional photos

Answer with ONLY one word from: chart, figure, scanned_text, miscellaneous
Do not include any explanation or thinking."""

VLM_CHART_CAPTION_PROMPT = """2枚の画像が提供されています。1枚目はページ全体、2枚目がキャプション対象のグラフです。

2枚目のグラフについて、すべての情報を含む詳細な説明文を日本語で書いてください。

説明文には以下を自然な文章で含めること：
- グラフの種類とタイトル
- 軸の名称と単位
- 凡例の項目名
- すべての数値データ（省略禁止）
- データの傾向や比較結果

出力は説明文のみ。箇条書きや見出し記号は使わず、流れるような段落形式で記述すること。指示文の繰り返しは禁止。日本語のみ。"""

VLM_FIGURE_CAPTION_PROMPT = """2枚の画像が提供されています。1枚目はページ全体、2枚目がキャプション対象の図/ダイアグラムです。

2枚目の図について、全体像を包括的に説明する文章を日本語で書いてください。

説明文には以下を自然な文章で含めること：
- 図の種類と全体的な目的・意味
- 図内のすべてのテキスト要素（ボックス、ラベル、注釈など）
- 要素間の関係性と流れ（矢印の方向、接続、階層）
- 色分けやグループ化の意味

重要：個々の要素を箇条書きで列挙するのではなく、図全体が何を伝えているかを統合的に説明すること。出力は説明文のみ。見出し記号や箇条書きは使わず、流れるような段落形式で記述すること。指示文の繰り返しは禁止。日本語のみ。"""

VLM_SCANNED_TEXT_CAPTION_PROMPT = """【指示】この画像に含まれるテキストをそのまま抽出してください。

【必須要件】
- 画像内のすべてのテキストを正確に書き起こすこと
- 元の構造・レイアウトをできる限り維持すること（段落、箇条書き、見出しなど）
- 表がある場合は、行と列の構造を維持して記載すること
- 読み取れない文字は「[判読不能]」と記載すること

【出力形式】
- テキストをそのまま出力（説明や解釈は不要）
- 見出しは「# 」で示す
- 箇条書きは「- 」で示す
- 段落の区切りは空行で示す

【厳守事項】
- テキスト以外の説明は不要（「この画像には〜」などの前置き禁止）
- テキストの内容を要約したり解釈したりしないこと
- 回答は日本語のみ（英語テキストもそのまま転記）"""

VLM_MISC_CAPTION_PROMPT = """【指示】2枚の画像が提供されています。1枚目はページ全体、2枚目が対象画像です。
2枚目の対象画像について、簡潔かつ情報密度の高いキャプションを日本語で生成してください。

【出力形式】以下の情報を1〜3文で簡潔にまとめてください：
- 画像の種類と主題（何の画像か）
- 画像内の重要なテキスト・ラベル（省略せず正確に）
- ページ上の関連する見出しやセクション名
- 写真の場合：人物名（判明している場合）、役職、状況

【厳守事項】
- 冗長な説明や繰り返しを避け、情報を凝縮すること
- 「この画像は〜」などの前置きは不要
- 推測ではなく確認できる情報のみ記述
- 回答は日本語のみ"""

VLM_JAPANESE_SYSTEM_MESSAGE = """あなたは日本語専門のアシスタントです。
すべての回答は必ず日本語のみで記述してください。
英語での回答は絶対に禁止です。
画像内に英語のテキストがあっても、説明は日本語で行ってください。"""
