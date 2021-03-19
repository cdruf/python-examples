\ Model giapetto_model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  3 x[0] + 2 x[1]
Subject To
 c[0]: 2 x[0] + x[1] <= 100
 c[1]: x[0] + x[1] <= 80
 c[2]: x[0] <= 40
Bounds
End
