"""
이커머스 정산 시스템 - pytest 테스트 스위트 (커버리지 80% 이상 보강판)
[AI 활용 CI/CD 교육] Day 1 · Part 4

실행:
  pytest src/tests/ -v --cov=src/settlement --cov-report=term-missing --cov-fail-under=80

■ 이번에 보강한 이유 (커버리지 78.40% → 80% 이상)
  main.py 커버리지 리포트에서 아래가 통째로 비어 있었습니다:
    - 108-114, 134-151 : lifespan/_seed_sample_data()
        → TestClient(app)를 컨텍스트 매니저로 쓰지 않아서
          FastAPI startup 이벤트(lifespan)가 실행된 적이 없었음.
          client fixture를 `with TestClient(app) as c: yield c`로 바꿔서 해결.
    - 321-329 : PUT /api/v1/orders/{id}/complete   (테스트 없음 → 추가)
    - 359     : GET /api/v1/orders                 (필터 없는 케이스 → 추가)
    - 402-412 : POST /api/v1/settlements            (테스트 없음 → 추가)
    - 473-479 : POST /api/v1/settlements/{id}/process (테스트 없음 → 추가)
  settlement_service.py:
    - 174-179 : get_orders() 자체가 테스트된 적이 없었음 → 서비스 레벨 테스트 추가

■ 의도적으로 남겨둔 미커버 구간 (실제 함수, Mock 없이는 도달 불가능한 방어 코드)
  - settlement_service.py 345-351 (process_settlement의 except 블록):
      현재 구현상 try 블록 안에서 실제로 예외를 던질 코드가 없어서
      (은행 API 호출이 주석처리된 교육용 코드) 정상 실행으로는
      도달할 수 없습니다. 강제로 도달시키려면 datetime.utcnow 같은
      내부 의존성을 monkeypatch해서 예외를 주입해야 하는데,
      이는 "실제 인스턴스 사용" 취지에서 벗어나는 방식이라 제외했습니다.
  - main.py 402-412 중 except 블록(500 응답):
      마찬가지로 svc.calculate_settlement()가 실패하는 상황을
      정상 입력으로는 만들 수 없어 제외했습니다.
  - models.py 42 (Order.amount_must_be_positive의 raise 줄):
      Order.amount는 이미 Field(..., ge=0) 제약이 있어서, 음수를 넣으면
      pydantic 코어 검증(ge=0)이 커스텀 field_validator보다 먼저 실패합니다.
      즉 이 줄은 사실상 도달 불가능한 죽은 코드입니다(ge=0과 중복).
      커버리지를 위해서라기보다, 코드 정리 관점에서 이 커스텀 validator
      자체를 제거하는 걸 고려해볼 만합니다.
  이 구간들을 빼고도 80% 기준은 충분히 넘습니다.
"""
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from settlement.main import app
from settlement.models.models import Order, OrderStatus, SettlementStatus
from settlement.services.settlement_service import SettlementService

# ── 픽스처 ────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    # 컨텍스트 매니저로 사용해야 FastAPI의 lifespan(startup/shutdown)이
    # 실제로 실행된다. 이게 없으면 _seed_sample_data()가 한 번도 안 돎.
    with TestClient(app) as c:
        yield c


@pytest.fixture
def svc():
    return SettlementService()


@pytest.fixture
def sample_order():
    return Order(
        order_id=f"TEST-{uuid.uuid4().hex[:6]}",
        merchant_id="M-TEST",
        customer_id="C-001",
        amount=Decimal("100000"),
        fee_rate=Decimal("0.03"),
    )


def unique_merchant(prefix: str = "M-API") -> str:
    """API 테스트 간 데이터가 섞이지 않도록 매번 고유한 merchant_id 생성"""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ── 모델 단위 테스트 ──────────────────────────────────────────────────

class TestOrderModel:
    def test_fee_amount(self):
        o = Order(order_id="T1", merchant_id="M", customer_id="C",
                  amount=Decimal("100000"))
        assert o.fee_amount == Decimal("3000")

    def test_net_amount(self):
        o = Order(order_id="T2", merchant_id="M", customer_id="C",
                  amount=Decimal("100000"))
        assert o.net_amount == Decimal("97000")

    def test_default_status_pending(self):
        o = Order(order_id="T3", merchant_id="M", customer_id="C",
                  amount=Decimal("50000"))
        assert o.status == OrderStatus.PENDING

    def test_negative_amount_raises(self):
        with pytest.raises(Exception):
            Order(order_id="T4", merchant_id="M", customer_id="C",
                  amount=Decimal("-1"))

    def test_fee_rounding(self):
        o = Order(order_id="T5", merchant_id="M", customer_id="C",
                  amount=Decimal("33333"), fee_rate=Decimal("0.03"))
        assert o.fee_amount == Decimal("1000")


# ── 서비스 단위 테스트 ────────────────────────────────────────────────

class TestSettlementService:
    def test_add_and_complete_order(self, svc, sample_order):
        svc.add_order(sample_order)
        done = svc.complete_order(sample_order.order_id)
        assert done is not None
        assert done.status == OrderStatus.COMPLETED
        assert done.completed_at is not None

    def test_complete_nonexistent_returns_none(self, svc):
        assert svc.complete_order("NONE-EXIST") is None

    def test_calculate_settlement_basic(self, svc):
        merchant = "M-CALC"
        amounts = [Decimal("50000"), Decimal("100000"), Decimal("200000")]
        for i, amt in enumerate(amounts):
            o = Order(order_id=f"O-{i}", merchant_id=merchant,
                      customer_id="C", amount=amt)
            svc.add_order(o)
            svc.complete_order(o.order_id)

        start = datetime.utcnow() - timedelta(hours=1)
        end   = datetime.utcnow() + timedelta(hours=1)
        rec   = svc.calculate_settlement(merchant, start, end)

        expected_sales = sum(amounts)
        expected_fee   = sum(a * Decimal("0.03") for a in amounts)

        assert rec.order_count  == 3
        assert rec.total_sales  == expected_sales
        assert rec.total_fee.quantize(Decimal("1")) == expected_fee.quantize(Decimal("1"))
        assert rec.net_amount   == expected_sales - rec.total_fee
        assert rec.status       == SettlementStatus.PENDING

    def test_pending_orders_excluded(self, svc):
        o = Order(order_id="PEND-1", merchant_id="M-X",
                  customer_id="C", amount=Decimal("100000"))
        svc.add_order(o)

        start = datetime.utcnow() - timedelta(hours=1)
        end   = datetime.utcnow() + timedelta(hours=1)
        rec   = svc.calculate_settlement("M-X", start, end)

        assert rec.order_count == 0
        assert rec.total_sales == Decimal("0")

    def test_calculate_settlement_no_orders_at_all(self, svc):
        """주문을 하나도 추가하지 않은 완전히 빈 상태에서 정산 계산."""
        rec = svc.calculate_settlement(
            merchant_id="M-EMPTY",
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow() + timedelta(days=30),
        )
        assert rec.order_count == 0
        assert rec.total_sales == Decimal("0")
        assert rec.total_fee == Decimal("0")
        assert rec.net_amount == Decimal("0")
        assert rec.status == SettlementStatus.PENDING
        assert rec.settlement_id.startswith("STL-")

    def test_process_settlement(self, svc, sample_order):
        svc.add_order(sample_order)
        svc.complete_order(sample_order.order_id)

        rec  = svc.calculate_settlement(
            "M-TEST",
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow() + timedelta(hours=1),
        )
        done = svc.process_settlement(rec.settlement_id)

        assert done.status == SettlementStatus.COMPLETED
        assert done.processed_at is not None

    def test_process_settlement_not_found_returns_none(self, svc):
        assert svc.process_settlement("STL-NOT-EXIST") is None

    def test_list_settlements_filter(self, svc):
        for m in ["M-A", "M-B"]:
            o = Order(order_id=f"O-{m}", merchant_id=m,
                      customer_id="C", amount=Decimal("10000"))
            svc.add_order(o)
            svc.complete_order(o.order_id)
            svc.calculate_settlement(
                m,
                datetime.utcnow() - timedelta(hours=1),
                datetime.utcnow() + timedelta(hours=1),
            )

        result = svc.list_settlements(merchant_id="M-A")
        assert all(r.merchant_id == "M-A" for r in result)

    def test_list_settlements_merchant_and_status_combined(self, svc):
        merchant = "M-COMBINED"
        for i in range(2):
            o = Order(order_id=f"CMB-{i}", merchant_id=merchant,
                      customer_id="C", amount=Decimal("10000"))
            svc.add_order(o)
            svc.complete_order(o.order_id)

        now = datetime.utcnow()
        rec_pending = svc.calculate_settlement(
            merchant, now - timedelta(hours=1), now + timedelta(hours=1)
        )
        rec_completed = svc.calculate_settlement(
            merchant, now - timedelta(hours=1), now + timedelta(hours=1)
        )
        svc.process_settlement(rec_completed.settlement_id)

        result = svc.list_settlements(
            merchant_id=merchant, status=SettlementStatus.COMPLETED
        )
        assert len(result) == 1
        assert result[0].settlement_id == rec_completed.settlement_id

        result_pending = svc.list_settlements(
            merchant_id=merchant, status=SettlementStatus.PENDING
        )
        assert len(result_pending) == 1
        assert result_pending[0].settlement_id == rec_pending.settlement_id

    def test_list_settlements_no_match_returns_empty(self, svc):
        result = svc.list_settlements(
            merchant_id="M-NEVER-EXISTS", status=SettlementStatus.COMPLETED
        )
        assert result == []

    def test_get_orders_filtered_by_merchant(self, svc):
        """get_orders(merchant_id=...) - 특정 판매자만 필터링되는 분기 커버"""
        svc.add_order(Order(order_id="GO-1", merchant_id="M-GET-A",
                             customer_id="C", amount=Decimal("1000")))
        svc.add_order(Order(order_id="GO-2", merchant_id="M-GET-B",
                             customer_id="C", amount=Decimal("2000")))

        result = svc.get_orders(merchant_id="M-GET-A")
        assert len(result) == 1
        assert result[0].order_id == "GO-1"

    def test_get_orders_without_filter_returns_all(self, svc):
        """get_orders() - merchant_id 없이 호출하면 전체 반환되는 분기 커버"""
        svc.add_order(Order(order_id="GO-3", merchant_id="M-GET-C",
                             customer_id="C", amount=Decimal("3000")))
        svc.add_order(Order(order_id="GO-4", merchant_id="M-GET-D",
                             customer_id="C", amount=Decimal("4000")))

        result = svc.get_orders()
        order_ids = {o.order_id for o in result}
        assert {"GO-3", "GO-4"}.issubset(order_ids)

        # list(self._orders) 복사본을 반환하므로 원본 훼손 방지 확인
        result.clear()
        assert len(svc.get_orders()) >= 2


# ── API 통합 테스트 ───────────────────────────────────────────────────

class TestAPI:
    def test_health(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["version"] == "1.0.0"
        assert "timestamp" in body

    def test_ready(self, client):
        res = client.get("/ready")
        assert res.status_code == 200
        assert res.json()["status"] == "ready"

    def test_create_order(self, client):
        payload = {
            "order_id": f"API-{uuid.uuid4().hex[:6]}",
            "merchant_id": "M-API",
            "customer_id": "C-001",
            "amount": "75000",
            "fee_rate": "0.03",
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
        }
        res = client.post("/api/v1/orders", json=payload)
        assert res.status_code == 201
        assert res.json()["order_id"] == payload["order_id"]

    def test_create_order_negative_amount_returns_422(self, client):
        payload = {
            "order_id": f"API-NEG-{uuid.uuid4().hex[:6]}",
            "merchant_id": "M-API",
            "customer_id": "C-001",
            "amount": "-1000",
        }
        res = client.post("/api/v1/orders", json=payload)
        assert res.status_code == 422

    def test_create_order_missing_required_field_returns_422(self, client):
        payload = {
            "order_id": f"API-MISS-{uuid.uuid4().hex[:6]}",
            "merchant_id": "M-API",
            "amount": "10000",
        }
        res = client.post("/api/v1/orders", json=payload)
        assert res.status_code == 422

    def test_complete_order_success(self, client):
        """PUT /api/v1/orders/{id}/complete 성공 케이스 (main.py 321-329 커버)"""
        order_id = f"API-COMPLETE-{uuid.uuid4().hex[:6]}"
        client.post("/api/v1/orders", json={
            "order_id": order_id,
            "merchant_id": "M-API",
            "customer_id": "C-001",
            "amount": "30000",
        })

        complete_res = client.put(f"/api/v1/orders/{order_id}/complete")
        assert complete_res.status_code == 200
        body = complete_res.json()
        assert body["status"] == "completed"
        assert body["completed_at"] is not None

    def test_complete_order_not_found_returns_404(self, client):
        """PUT .../complete 404 케이스 (main.py 321-329의 raise 분기 커버)"""
        res = client.put("/api/v1/orders/ORD-NOT-EXIST/complete")
        assert res.status_code == 404
        assert "ORD-NOT-EXIST" in res.json()["detail"]

    def test_list_orders_filter_by_merchant(self, client):
        merchant = unique_merchant("M-LIST")
        order_id = f"ORD-{uuid.uuid4().hex[:6]}"
        client.post("/api/v1/orders", json={
            "order_id": order_id,
            "merchant_id": merchant,
            "customer_id": "C-001",
            "amount": "20000",
        })

        res = client.get(f"/api/v1/orders?merchant_id={merchant}")
        assert res.status_code == 200
        orders = res.json()
        assert len(orders) == 1
        assert orders[0]["order_id"] == order_id

    def test_list_orders_without_filter(self, client):
        """GET /api/v1/orders (필터 없음) - main.py 359 라인 커버.
        lifespan에서 시딩된 데이터가 있어 최소 1건 이상이어야 한다.
        """
        client.post("/api/v1/orders", json={
            "order_id": f"ORD-{uuid.uuid4().hex[:6]}",
            "merchant_id": unique_merchant("M-ANY"),
            "customer_id": "C-001",
            "amount": "5000",
        })
        res = client.get("/api/v1/orders")
        assert res.status_code == 200
        assert isinstance(res.json(), list)
        assert len(res.json()) >= 1

    def test_list_settlements(self, client):
        res = client.get("/api/v1/settlements")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_list_settlements_filter(self, client):
        res = client.get("/api/v1/settlements?merchant_id=M-001")
        assert res.status_code == 200

    def test_create_settlement_via_api(self, client):
        """POST /api/v1/settlements 성공 케이스 (main.py 402-412 커버)"""
        merchant = unique_merchant("M-SETTLE")
        order_id = f"ORD-{uuid.uuid4().hex[:6]}"

        client.post("/api/v1/orders", json={
            "order_id": order_id,
            "merchant_id": merchant,
            "customer_id": "C-001",
            "amount": "40000",
        })
        client.put(f"/api/v1/orders/{order_id}/complete")

        now = datetime.utcnow()
        settlement_payload = {
            "merchant_id": merchant,
            "period_start": (now - timedelta(hours=1)).isoformat(),
            "period_end": (now + timedelta(hours=1)).isoformat(),
        }
        res = client.post("/api/v1/settlements", json=settlement_payload)
        assert res.status_code == 201
        body = res.json()
        assert body["merchant_id"] == merchant
        assert body["order_count"] == 1
        assert body["status"] == "pending"

    def test_settlements_merchant_and_status_filter_via_api(self, client):
        merchant = unique_merchant("M-FILTER")
        order_id = f"ORD-{uuid.uuid4().hex[:6]}"

        client.post("/api/v1/orders", json={
            "order_id": order_id,
            "merchant_id": merchant,
            "customer_id": "C-001",
            "amount": "15000",
        })
        client.put(f"/api/v1/orders/{order_id}/complete")

        now = datetime.utcnow()
        create_res = client.post("/api/v1/settlements", json={
            "merchant_id": merchant,
            "period_start": (now - timedelta(hours=1)).isoformat(),
            "period_end": (now + timedelta(hours=1)).isoformat(),
        })
        settlement_id = create_res.json()["settlement_id"]

        pending_res = client.get(
            f"/api/v1/settlements?merchant_id={merchant}&status=pending"
        )
        assert pending_res.status_code == 200
        assert len(pending_res.json()) == 1

        completed_before = client.get(
            f"/api/v1/settlements?merchant_id={merchant}&status=completed"
        )
        assert completed_before.json() == []

        # POST /api/v1/settlements/{id}/process 성공 케이스 (main.py 473-479 커버)
        process_res = client.post(f"/api/v1/settlements/{settlement_id}/process")
        assert process_res.status_code == 200
        assert process_res.json()["status"] == "completed"

        completed_after = client.get(
            f"/api/v1/settlements?merchant_id={merchant}&status=completed"
        )
        assert len(completed_after.json()) == 1
        assert completed_after.json()[0]["settlement_id"] == settlement_id

        pending_after = client.get(
            f"/api/v1/settlements?merchant_id={merchant}&status=pending"
        )
        assert pending_after.json() == []

    def test_process_settlement_not_found_returns_404(self, client):
        """POST .../process 404 케이스 (main.py 473-479의 raise 분기 커버)"""
        res = client.post("/api/v1/settlements/STL-NOT-EXIST/process")
        assert res.status_code == 404
        assert "STL-NOT-EXIST" in res.json()["detail"]

    def test_list_settlements_invalid_status_returns_422(self, client):
        res = client.get("/api/v1/settlements?status=invalid_status")
        assert res.status_code == 422
