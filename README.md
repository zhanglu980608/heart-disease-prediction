# heart-disease-prediction

## Style

```{r}
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse, ggtext
)

font_add_google(name = "Roboto Mono", family = "Roboto Mono")

theme_set(
  theme_minimal() +
    theme(
      legend.position = "bottom",
      legend.background = element_rect(fill = "#F9EFE6", color = "#F9EFE6"),
      legend.key = element_rect(fill = "#F9EFE6", color = "#F9EFE6"),
      legend.title = element_text(size = 7, color = "#3B372E"),
      legend.text = element_text(size = 7, color = "#3B372E"),
      plot.background = element_rect(fill = "#F9EFE6", color = "#F9EFE6"),
      panel.background = element_rect(fill = "#F9EFE6", color = "#F9EFE6"),
      text = element_text(
        family = "Roboto Mono",
        color = "#3B372E"
      ),
      axis.title.y = element_text(vjust = 0.2, face = "bold"),
      axis.title.x = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(),
      axis.text.y = element_text(angle = 30),
      plot.title = element_markdown(
        size = 18, hjust = 0.5,
        family = "Roboto Slab"
      ),
      plot.subtitle = element_markdown(
        hjust = 0.5,
        family = "Roboto Slab"
      ),
      plot.caption = element_markdown(
        size = 10,
        family = "Roboto Slab",
        hjust = 0
      ),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(15, 15, 15, 15)
    )
)
```