# 1. Decision Tree

- Supervised Learning 알고리즘 중 하나이다.
- 데이터 마이닝, 머신러닝에서 자주 사용하는 알고리즘이다.



# 2. Algorithm

>  가정
>
>  1. Train Data : ${(x_1, y_1), (x_2, y_2), (x_3, y_3), ..., (x_n, y_n)}$
>  2. $x_i$ : $k$개의 feature가 있는 $i$ 번째 데이터 샘플
>  3. $y_i$ : $i$ 번째 데이터 샘플의 클래스

**Step 1. 분기 전 Entropy, 분기 후 Entropy 계산**

**Step 2. 분기 후보들의 Information Gain 계산**

**Step 3. 가장 높은 Information Gain 의 분기 선택**

**Step 4. 위 과정을 종료 조건에 만족할 때까지 반복**



# 3. Entropy, Information Gain

- Entropy : 데이터가 균일하게 분류되어 있는지에 대한 척도

  - 분기 전의 Entropy 계산 공식

    <img src="../images/Entropy formula.png">

  - 분기 후의 Entropy 계산 공식

    <img src="../images/Entropy formula2.png">

  - Entropy 大 : 클래스의 분포가 균등하다.

  - Entropy 小 : 클래스의 분포가 불균등하다. (한쪽으로 기울어져 분류된 상태)

- Information Gain : 분기 이전의 Entropy와 분기 이후의 Entropy의 차이

  <img src="../images/Information Gain formula.png">

  - Information Gain 大 : 현재 선택한 분기에서 클래스의 분포가 불균형하다. 잘 분리되었다라고 판단.

- Entropy의 그래프는 다음과 같다.

  <img src="../images/Entropy graph.png">

  - 클래스의 분포가 균등하게 나와 각 클래스의 확률이 같다면 $-0.5log_20.5 -0.5log_20.5 = 1$ 로 최댓값을 계산할 수 있다.
  - 반대로 클래스의 분포가 하나의 클래스로만 분류되었다면 $-1log_21 -0log_20 = 0$, $\displaystyle\lim_{x\rightarrow0}xlogx = 0$ 로 최솟값을 계산할 수 있다.



# 4. Decision Tree의 장단점

- 장점
  1. 해석하기 쉽다. 즉, 해당 클래스로 예측했을 때 트리를 따라가며 예측 과정을 해석할 수 있다.
  2. 구현과 이해하기 쉬운 모델이다.
  3. 비모수적 모델이다. 즉, 통계모델에 요구되는 가정이 없음 (정규성, 독립성, 등분산성 등..)
- 단점
  1. 데이터의 수가 적을수록 모델이 불안정하다.
  2. Overfitting 발생 확률이 높다.


# 5. Decision Tree 예제

다음과 같은 데이터 테이블이 있을때, Decision Tree를 만들어 Football을 할 수 있는지 없는지를 예측하는 모델을 만들고자 한다.

| Outlook  | Temperature | Humidity | Wind   | Played Football |
| -------- | ----------- | -------- | ------ | :-------------: |
| Sunny    | Hot         | High     | Weak   |        N        |
| Sunny    | Hot         | High     | Strong |        N        |
| Overcast | Hot         | High     | Weak   |        Y        |
| Rain     | Mild        | High     | Weak   |        Y        |
| Rain     | Cool        | Normal   | Weak   |        Y        |
| Rain     | Cool        | Normal   | Strong |        N        |
| Overcast | Cool        | Normal   | Strong |        Y        |
| Sunny    | Mild        | High     | Weak   |        N        |
| Sunny    | Cool        | Normal   | Weak   |        Y        |
| Rain     | Mild        | Normal   | Weak   |        Y        |
| Sunny    | Mild        | Normal   | Strong |        Y        |
| Overcast | Mild        | High     | Strong |        Y        |
| Overcast | Hot         | Normal   | Weak   |        Y        |
| Rain     | Mild        | High     | Strong |        N        |

첫번째 분기를 다음과 같이 결정한다.

- 분기전 Entropy를 다음과 같이 계산 할 수 있다.

  - 분기전의 클래스 분포는 다음과 같고,

    | Played  Football |      |
    | ---------------- | ---- |
    | N                | 5    |
    | Y                | 9    |

  - 분기전의 Entropy는 다음과 같이 계산 할 수 있다.

    $E(S) = -[(\frac{9}{14})log(\frac{9}{14}) + (\frac{5}{14})log(\frac{5}{14})] = 0.94$

- 분기가 될 후보들은 다음과 같다.

  - **Outlook, Temperature, Humidity, Wind**

  - 각 분기의 클래스의 분포와 Entropy, Information Grain을 다음과 같이 계산 할 수 있다.

    - outlook

      |         |          |      | play |        |
      | ------- | -------- | ---- | ---- | :----: |
      |         |          | yes  | no   | total  |
      |         | sunny    | 3    | 2    |   5    |
      | outlook | overcast | 4    | 0    |   4    |
      |         | rainy    | 2    | 3    |   5    |
      |         |          |      |      | **14** |

      $E(S, outlook) = \frac{5}{14}\times E(3,2) + \frac{4}{14}\times E(4,0) + \frac{5}{14}\times E(2,3)$

      ​                           $= \frac{5}{14}[-\frac{3}{5}log(\frac{3}{5})-\frac{2}{5}log(\frac{2}{5})]+ \frac{4}{14}[0] + \frac{5}{14}[\frac{2}{5}log(\frac{2}{5})-\frac{3}{5}log(\frac{3}{5})] = 0.693$

      $IG(S, outlook) = 0.94 - 0.693 = 0.247$

    - temperature

      |             |      |      | play |       |
      | ----------- | ---- | ---- | ---- | :---: |
      |             |      | yes  | no   | total |
      |             | Hot  | 2    | 2    |   4   |
      | Temperature | Mild | 4    | 2    |   6   |
      |             | Cool | 3    | 1    |   4   |
      |             |      |      |      |  14   |

      $E(S, Temperature) = 0.911$

      $IG(S, Temperature) = 0.94 - 0.911 = 0.029$

    - humidity

      |          |        |      | play |       |
      | -------- | ------ | ---- | ---- | :---: |
      |          |        | yes  | no   | total |
      |          | High   | 3    | 4    |   7   |
      | Humidity | Normal | 6    | 1    |   7   |
      |          |        |      |      |  14   |

      $E(S, Humidity) = 0.788$

      $IG(S, Humidity) = 0.94 - 0.788 = 0.152$

    - wind

      |      |        |      | play |       |
      | ---- | ------ | ---- | ---- | :---: |
      |      |        | yes  | no   | total |
      |      | Strong | 3    | 3    |   6   |
      | Wind | Weak   | 6    | 2    |   8   |
      |      |        |      |      |  14   |

      $E(S, Windy) = 0.893$

      $IG(S, Windy) = 0.94 - 0.893 = 0.047$

    - Information Gain이 가장 높은 **Outlook**을 선택하고 분기시킨다.

      <img src="../images/decision tree first split.png">

      분기된 클래스의 분포를 각 특징에 대해서 비교 했을 때, Overcast의 경우 모두 y로 분리 되어 더 이상 분기를 할 필요없게 되었다.

      <img src="../images/decision tree first split2.png" width="200">

이제 두번째 분기는 다음과 같이 결정된다.

- Sunny를 기준으로 분기

  - $E(sunny) = [\frac{3}{5}log(3/5)-\frac{2}{5}log(2/5)] = 0.971$

  - 분기 후보는 **Temperature, Humidity, Windy**이다. 각 후보의 Entropy와 Information Gain을 계산하면 다음과 같다.

    - Temperature

      $E(sunny, Temperature) = \frac{2}{5}*E(0,2) + \frac{2}{5}*E(1,1) + \frac{1}{5}*E(1,0)=\frac{2}{5}=0.4$

      $IG(sunny, Temperature) = 0.971–0.4 =0.571$

    - Humidity

      $E(sunny, Humidity)=0$

      $IG(sunny, Humidity) = 0.971-0=0.971$

    - Windy

      $E(sunny, Windy)=0.951$

      $IG(sunny, Windy) = 0.971-0.951=0.020$

  - 가장 Information Gain을 가지는 Humidity로 분기한다.

    <img src="../images/decision tree second split.png", width="700">

    ​

- Rain을 기준으로 분기했을 때 위와 같이 계산하면 Wind를 선택하게 되고 다음과 같이 Decision Tree를 완성 할 수 있다.

  <img src="../images/decision tree complete.png", width="700">

- 실제 데이터가 다음과 같이 주어졌을 때 어떻게 분류되는지 살펴보자

  - $X_{data}: Outlook = Sunny, Temperature = Cool, Humidity = Normal, Wind = Strong$
  - **Outlook = Sunny** and **Humidity = Normal**로 풋볼을 하게되는 결과를 얻을 것이다.

- 규칙 추출 (5개의 규칙 생성, 5개의 leaf node)

  1. IF (Outlook = Sunny) and (Humidity = High) THEN Football = No
  2. IF (Outlook = Sunny) and (Humidity = Normal) THEN Football = Yes
  3. IF (Outlook = Rain) and (Wind = Strong) THEN Football = No
  4. IF (Outlook = Rain) and (Wind = Weak) THEN Football = Yes
  5. IF (Outlook = Overcast) THEN Football = Yes

# 6.Decision Tree의 종류

- ID3 (Iterative Dichotomiser 3)
  - Entropy와 Information Gain 지표를 사용
- CART (Classification and Regression Trees)
  - gini-impurity를 사용

## Reference :

- [Decision Tree Algorithm With Hands-On Example](https://medium.datadriveninvestor.com/decision-tree-algorithm-with-hands-on-example-e6c2afb40d38)