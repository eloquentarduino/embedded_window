# Sliding window for C++ projects

This package implements a sliding window that computes basic features (min/max/mean/std...) to use
in Machine Learning projects and that is able to export itself to plain C++.

Main intended audience is the Arduino community, but the code is very general-purpose and can be adapted
to many scenarios.

For a demo, visit [the Jupyter Notebook](https://github.com/eloquentarduino/user-projects/blob/main/Covid%20Patient%20Health%20Assessing%20Device%20Using%20Sliding%20Window.ipynb).

## Install

```bash
pip install embedded_window
```

## Use in Python

```python
from embedded_window import Window

window = Window(length=20, shift=5)

# X_w holds the input arranged in windows. Shape is (-1, length, X.shape[1])
# features holds the extracted features for each window (min/max/mean/std...)
# y_w holds the most frequent label inside each window 
X_w, features, y_w = window.fit_transform(X, y)

# export to C++
print(window.port())
```

## Use in Arduino C++

```cpp
#include "Window.h"


Window window;
float X[30][4] = {...};


void setup() {
  Serial.begin(115200);
  delay(2000);
}

void loop() {
  for (int i = 0; i < 30; i++) {
    if (window.transform(X[i])) {
      print_array(window.features, window.features_count);
    }
  }

  delay(60000);
}
```