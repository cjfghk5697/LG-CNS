# LG-CNS-

## 6/30 제출본 참고사항
1. 배달원 종류를 배치할 때 번들 하나씩 변경 가능, 불가능 여부를 보는 대신에 전체 배달원을 전부 최적으로 재배치하는 방식으로 개선하였다.
- 실험 결과 번들내 주문 상태가 전부 정해진 상태에서 CAR보다는 BIKE나 WALK 배달원을 배치하는게 비용이 가장 낮았다. 또한 BIKE는 전부 배치되어 있는 경우가 많았으며 BIKE의 속도가 가장 빠르기 때문으로 보인다. 반대로, CAR와 WALK 배달원은 항상 남아있었다.
- 이를 다음과 같이 구현하였다.
- 1. 번들 중 한 종류의 배달원으로만 가능하거나 CAR가 최선이면 바로 배치한다.
- 2. WALK가 비용이 가장 낮은 번들에 WALK 배달원을 할당한다. 만약 WALK 배달원의 수가 부족한 경우 두 번째로 비용이 적게 드는 배달원 종류를 배치했을 때와의 비용 차이가 가장 큰 번들부터 WALK 배달원을 할당한다.
- 3. BIKE가 비용이 가장 낮은 번들에 BIKE 배달원을 할당하며, BIKE 배달원이 부족한 경우 CAR 배달원하고의 비용 차이가 가장 큰 번들부터 할당한다.
- 4. 남은 배달원 중 가장 적은 비용을 가진 배달원 종류를 바로 할당한다.

3. 주문이 4개 묶인 번들도 가능하며, 이것이 많을수록 총 비용이 내려간다.
- 최대한 번들에 많은 주문씩 묶을수록 총 비용이 내려가는 것을 확인하였다.
- 또한 번들이 4개까지 묶이는 경우도 가능함을 확인하였으며, 번들을 최대한 주문 4개까지 묶을 수 있게 만들었다.
- 구체적으로, 먼저 번들을 최대 주문 2개씩 묶은 다음, 이후 번들을 최대 4개씩 묶고서 마지막으로 주문 2개 이하로 묶인 번들을 모두 풀고서 다시 주문 3개까지 묶을 수 있게 하였다.
- 제출 결과 Time Limit이 나왔는데, 6/29 제출본에서의 최적 가중치 탐색 그리드 서치 방식에서 시간이 너무 많이 쓰인 것 같다.


## 6/29 제출본 참고사항
- 출발지 사이의 거리, 도착지 사이의 거리 외에도 주문 ready time의 차이, 마감 시간의 차이, 출발지와 도착지 사이의 평균 거리 또한 가중치를 임의로 주어 반영하였다.
- 이때 적절한 가중치를 탐색하는 2중 for문을 이용하였으며, K=300일 때 4초 정도 소요된다.


## 6/28 제출본 참고사항

### 코드 구조의 특징
- util 대신 일부 코드를 개선하여 '함수명2'와 같은 형태로 만든 util2 모듈을 이용하였다.

### Bassline 코드 실험으로 얻은 아이디어
- solution의 번들 정보를 확인해본 결과 주문은 최대 3개까지만 묶이며 4개까지 묶인 번들이 하나라도 있는 경우는 드물었다. (예시로, Bassline 결과 하나의 '번들 내 주문 수: 해당 빈도'는 다음과 같았다. '1: 25, 2: 65, 3: 15') -> 그러므로 번들은 주문 3개까지 묶는 경우만 고려해도 될 수 있다고 생각하였다.

### 기본 가정
- 총 비용이 적은 번들 상태에서 최적화를 시작하는 것이 대부분의 경우에 유리할 것이라고 생각한다. -> 비용이 적은 곳에서 출발하니까 오히려 local minimum에 갇히는 현상이 나타나지 않음을 가정한다.

### 주요 아이디어
- 번들을 묶을 때 출발지 사이가 서로 가깝고, 도착지 사이가 서로 가까운 주문끼리 묶을 수 있다면 해당 주문의 비용을 줄일 수 있을 것이다.
- 주문을 하나씩만 포함한 모든 번들의 모든 쌍을 이용해서, 만약 a와 b 번들이 있다고 할 때 그 둘이 각각 가진 주문의 (출발지 사이의 거리 + 도착지 사이의 거리)를 (a, b, dist)와 같은 튜플 형태로 저장한 다음 dist의 오름차순으로 정렬해서 거리가 가까운 조합을 먼저 탐색하도록 한다.
- 이후 모든 쌍을 탐색하면서 두 번들을 try_merging_bundles로 합칠 수 있다면 합치고 union find를 이용하여 합쳐진 번들의 위치를 효율적으로 기억한다.
- try_merging_bundles는 모든 라이더의 모든 출발지, 도착지 순열을 확인할 수 있게 수정하였다. 번들 내의 주문을 최대 3개까지만 허용한다면, 이러한 경우에도 충분히 빠른 속도가 나온다.

### 추가 작업
- draw_route_solution2는 draw_route_solution의 시각화 결과에서 주문 번호가 같이 보이도록 수정하였다.

### 기타 코멘트
- DIST 행렬 대신에 주문 좌표로 직접 거리를 계산해본 결과 유클리디언 거리 방식이 가장 근접하지만 정확히 같지는 않았다. 혹시라도 좌표를 사용하여 평균을 낼 필요가 있을지 몰라서 좌표를 통해 거리를 직접 계산하는 방식으로 해보았다.
- 구현 방식에 최소 스패닝 트리의 크루스칼 알고리즘을 응용하였다.
