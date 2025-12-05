from fastapi import FastAPI, HTTPException, Depends, Body, status
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime
import contextlib
import uuid

SQLITE_DATABASE_URL = "sqlite:///./bookshop.db"

engine = create_engine(
    SQLITE_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Address(Base):
    __tablename__ = "address"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    country = Column(String(255), nullable=False)
    city = Column(String(255), nullable=False)
    street = Column(String(255), nullable=False)


class Client(Base):
    __tablename__ = "client"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    surname = Column(String(100), nullable=False)
    email = Column(String(255))
    address_id = Column(String, ForeignKey("address.id", ondelete="SET NULL"))
    registration_date = Column(DateTime, default=datetime.utcnow)


class Supplier(Base):
    __tablename__ = "supplier"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    phone = Column(String(20))
    address_id = Column(String, ForeignKey("address.id", ondelete="SET NULL"))


class Book(Base):
    __tablename__ = "book"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    author = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(Float, nullable=False)
    genre = Column(String(100))
    stock = Column(Integer, default=0)
    supplier_id = Column(String, ForeignKey("supplier.id", ondelete="SET NULL"))


class Image(Base):
    __tablename__ = "image"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    book_id = Column(String, ForeignKey("book.id", ondelete="CASCADE"), nullable=False)
    data = Column(Text, nullable=False)  # base64 строка или URL


Base.metadata.create_all(bind=engine)


class AddressCreate(BaseModel):
    country: str
    city: str
    street: str


class ClientBase(BaseModel):
    name: str
    surname: str
    email: Optional[str] = None


class ClientCreate(ClientBase):
    address: Optional[AddressCreate] = None


class ClientResponse(ClientBase):
    id: str
    registration_date: datetime
    address_id: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class SupplierCreate(BaseModel):
    name: str
    phone: Optional[str] = None
    address: Optional[AddressCreate] = None


class SupplierResponse(SupplierCreate):
    id: str
    address_id: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class BookBase(BaseModel):
    title: str
    author: str
    description: Optional[str] = None
    price: float
    genre: Optional[str] = None
    stock: int = 0
    supplier_id: Optional[str] = None


class BookCreate(BookBase):
    pass


class BookResponse(BookBase):
    id: str
    model_config = ConfigDict(from_attributes=True)


class StockDecreaseRequest(BaseModel):
    amount: int


class ImageCreate(BaseModel):
    book_id: str
    data: str


class ImageResponse(ImageCreate):
    id: str
    model_config = ConfigDict(from_attributes=True)


def add_test_data():
    db = SessionLocal()
    try:
        if db.query(Book).count() == 0:
            test_books = [
                Book(
                    title="1984",
                    author="Джордж Оруэлл",
                    description="Антиутопический роман о тоталитарном обществе.",
                    price=599.99,
                    genre="Антиутопия",
                    stock=10
                ),
                Book(
                    title="Мастер и Маргарита",
                    author="Михаил Булгаков",
                    description="Философский роман с элементами сатиры и мистики.",
                    price=699.99,
                    genre="Классика",
                    stock=5
                ),
                Book(
                    title="Гарри Поттер и Философский камень",
                    author="Дж. К. Роулинг",
                    description="Первая книга о приключениях юного волшебника.",
                    price=799.99,
                    genre="Фэнтези",
                    stock=15
                )
            ]
            db.add_all(test_books)
            db.commit()
            print("✅ Тестовые книги добавлены")
    except Exception as e:
        print(f"Ошибка при добавлении тестовых данных: {e}")
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    add_test_data()
    yield


app = FastAPI(
    title="Книжный магазин",
    lifespan=lifespan
)



@app.post("/clients/", response_model=ClientResponse, status_code=status.HTTP_201_CREATED)
def create_client(client: ClientCreate, db: Session = Depends(get_db)):
    address_id = None
    if client.address:
        addr = Address(**client.address.model_dump())
        db.add(addr)
        db.commit()
        db.refresh(addr)
        address_id = addr.id

    db_client = Client(
        name=client.name,
        surname=client.surname,
        email=client.email,
        address_id=address_id
    )
    db.add(db_client)
    db.commit()
    db.refresh(db_client)
    return db_client


@app.delete("/clients/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_client(client_id: str, db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Клиент не найден")
    db.delete(client)
    db.commit()


@app.get("/clients/", response_model=List[ClientResponse])
def get_clients(limit: Optional[int] = None, offset: Optional[int] = None, db: Session = Depends(get_db)):
    query = db.query(Client)
    if limit is not None and offset is not None:
        query = query.limit(limit).offset(offset)
    return query.all()


@app.get("/clients/search/", response_model=List[ClientResponse])
def search_clients(name: str, surname: str, db: Session = Depends(get_db)):
    clients = db.query(Client).filter(Client.name == name, Client.surname == surname).all()
    if not clients:
        raise HTTPException(status_code=404, detail="Клиенты не найдены")
    return clients


@app.put("/clients/{client_id}/address", status_code=status.HTTP_204_NO_CONTENT)
def update_client_address(client_id: str, address: AddressCreate, db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Клиент не найден")

    if client.address_id:
        old_addr = db.query(Address).filter(Address.id == client.address_id).first()
        if old_addr:
            db.delete(old_addr)

    new_addr = Address(**address.model_dump())
    db.add(new_addr)
    db.commit()
    db.refresh(new_addr)

    client.address_id = new_addr.id
    db.commit()



@app.post("/suppliers/", response_model=SupplierResponse, status_code=status.HTTP_201_CREATED)
def create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    address_id = None
    if supplier.address:
        addr = Address(**supplier.address.model_dump())
        db.add(addr)
        db.commit()
        db.refresh(addr)
        address_id = addr.id

    db_supplier = Supplier(
        name=supplier.name,
        phone=supplier.phone,
        address_id=address_id
    )
    db.add(db_supplier)
    db.commit()
    db.refresh(db_supplier)
    return db_supplier


@app.put("/suppliers/{supplier_id}/address", status_code=status.HTTP_204_NO_CONTENT)
def update_supplier_address(supplier_id: str, address: AddressCreate, db: Session = Depends(get_db)):
    supplier = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Поставщик не найден")

    if supplier.address_id:
        old_addr = db.query(Address).filter(Address.id == supplier.address_id).first()
        if old_addr:
            db.delete(old_addr)

    new_addr = Address(**address.model_dump())
    db.add(new_addr)
    db.commit()
    db.refresh(new_addr)

    supplier.address_id = new_addr.id
    db.commit()


@app.delete("/suppliers/{supplier_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_supplier(supplier_id: str, db: Session = Depends(get_db)):
    supplier = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Поставщик не найден")
    db.delete(supplier)
    db.commit()


@app.get("/suppliers/", response_model=List[SupplierResponse])
def get_suppliers(db: Session = Depends(get_db)):
    return db.query(Supplier).all()


@app.get("/suppliers/{supplier_id}", response_model=SupplierResponse)
def get_supplier(supplier_id: str, db: Session = Depends(get_db)):
    supplier = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Поставщик не найден")
    return supplier



@app.post("/books/", response_model=BookResponse, status_code=status.HTTP_201_CREATED)
def create_book(book: BookCreate, db: Session = Depends(get_db)):
    db_book = Book(**book.model_dump())
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    return db_book


@app.put("/books/{book_id}/decrease-stock", response_model=BookResponse)
def decrease_stock(book_id: str, request: StockDecreaseRequest, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")
    if book.stock < request.amount:
        raise HTTPException(status_code=400, detail="Недостаточно книг на складе")
    book.stock -= request.amount
    db.commit()
    db.refresh(book)
    return book


@app.get("/books/{book_id}", response_model=BookResponse)
def get_book(book_id: str, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")
    return book


@app.get("/books/", response_model=List[BookResponse])
def get_books(db: Session = Depends(get_db)):
    return db.query(Book).all()


@app.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_book(book_id: str, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")
    db.delete(book)
    db.commit()



@app.post("/images/", response_model=ImageResponse, status_code=status.HTTP_201_CREATED)
def add_image(image: ImageCreate, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.id == image.book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")

    db_image = Image(**image.model_dump())
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image


@app.put("/images/{image_id}", response_model=ImageResponse)
def update_image(image_id: str, image: ImageCreate, db: Session = Depends(get_db)):
    db_image = db.query(Image).filter(Image.id == image_id).first()
    if not db_image:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    db_image.data = image.data
    db.commit()
    db.refresh(db_image)
    return db_image


@app.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(image_id: str, db: Session = Depends(get_db)):
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    db.delete(image)
    db.commit()


@app.get("/images/by-book/{book_id}", response_model=List[ImageResponse])
def get_images_by_book(book_id: str, db: Session = Depends(get_db)):
    images = db.query(Image).filter(Image.book_id == book_id).all()
    if not images:
        raise HTTPException(status_code=404, detail="Изображения не найдены")
    return images


@app.get("/images/{image_id}", response_model=ImageResponse)
def get_image(image_id: str, db: Session = Depends(get_db)):
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    return image


@app.get("/")
def read_root():
    return {"message": "📚 Книжный магазин работает!"}


# Запуск через python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)