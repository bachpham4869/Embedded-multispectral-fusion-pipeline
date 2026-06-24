# Auto-Label Class Prompt Mapping

These prompts are for a future independent teacher path. In the current phase, they are documentation only.

| ENV class | Prompt | Caveat |
| --- | --- | --- |
| night_clear | clear outdoor night scene with very low ambient light | Standard taxonomy mapping. |
| normal_night | low-light night scene with ambient street or urban lighting | Standard taxonomy mapping. |
| normal_day | normal daylight outdoor scene | Standard taxonomy mapping. |
| fog | foggy or hazy low-visibility outdoor scene | Standard taxonomy mapping. |
| rain | rainy outdoor scene, wet road, wet lens, or visible rain streaks | Standard taxonomy mapping. |
| glare | direct glare from bright light source such as sun or headlights | Standard taxonomy mapping. |
| backlight | backlit scene with subject or foreground against strong background light | Standard taxonomy mapping. |
| transition | dawn or dusk transition lighting, not just model uncertainty | Use only for true dawn/dusk transient lighting, not uncertainty. |
| nir_night | active infrared or NIR monochrome night scene | Low confidence unless modality is confirmed NIR/IR-like. |
| unknown_or_out_of_scope | uncertain, ambiguous, indoor, non-environmental, or outside the taxonomy | Allowed when no class is reliable. |
