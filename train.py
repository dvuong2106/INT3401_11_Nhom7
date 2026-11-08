from torch import no_grad
from torch.utils.data import DataLoader


from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


def train_perceptron(model, dataset):
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        "*** YOUR CODE HERE ***"
        while True:
            errors = 0
            for batch in dataloader:
                x = batch['x']
                y = batch['label']

                if model.get_prediction(x) != int(y.item()):
                    model.w.add_(y * x)
                    errors += 1
            if errors == 0:
                break


def train_regression(model, dataset):
    "*** YOUR CODE HERE ***"
    model.train()
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(3000):
        for batch in dataloader:
            x = batch['x']
            y = batch['label']
            optimizer.zero_grad()
            prediction = model(x)
            loss = regression_loss(prediction, y)
            loss.backward()
            optimizer.step()

        with no_grad():
            x_all = tensor(dataset.x, dtype=x.dtype)
            y_all = tensor(dataset.y, dtype=y.dtype)
            current_loss = regression_loss(model(x_all), y_all).item()

        if current_loss <= 0.01:
            break

def train_digitclassifier(model, dataset):
    model.train()
    """ YOUR CODE HERE """
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(15):
        model.train()

        for batch in dataloader:
            optimizer.zero_grad()

            predictions = model(batch['x'])
            loss = digitclassifier_loss(predictions, batch['label'])

            loss.backward()
            optimizer.step()

        model.eval()
        with no_grad():
            validation_accuracy = dataset.get_validation_accuracy()

        print(
            f"Epoch {epoch + 1}: "
            f"validation accuracy = {validation_accuracy:.2%}"
        )

        if validation_accuracy >= 0.98:
            break

def train_languageid(model, dataset):
    model.train()
    "*** YOUR CODE HERE ***"
    batch_size = 64
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(22):
        model.train()

        bucket_loader = DataLoader(
            dataset.train_buckets,
            batch_size=1,
            shuffle=True
        )

        for bucket in bucket_loader:
            start, end = bucket[0].tolist()
            start, end = int(start), int(end)

            for i in range(start, end, batch_size):
                j = min(i + batch_size, end)

                if j - i == 1:
                    indices = [i, i]
                    xs, labels = dataset._encode(
                        dataset.train_x[indices],
                        dataset.train_y[indices]
                    )
                else:
                    xs, labels = dataset._encode(
                        dataset.train_x[i:j],
                        dataset.train_y[i:j]
                    )

                optimizer.zero_grad()

                predictions = model(xs)
                loss = languageid_loss(predictions, labels)

                loss.backward()
                optimizer.step()

        model.eval()

        with no_grad():
            accuracy = dataset.get_validation_accuracy()

        print(
            f"Epoch {epoch + 1:02d}: "
            f"accuracy = {accuracy:.2%}"
        )

        if accuracy >= 0.85:
            break


def Train_DigitConvolution(model, dataset):
    """ YOUR CODE HERE """
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.003
    )

    for epoch in range(35):
        model.train()

        for batch in dataloader:
            optimizer.zero_grad()

            predictions = model(batch['x'])
            loss = digitconvolution_Loss(
                predictions,
                batch['label']
            )

            loss.backward()
            optimizer.step()

        model.eval()

        with no_grad():
            accuracy = dataset.get_validation_accuracy()

        print(
            f"Epoch {epoch + 1:02d}: "
            f"validation accuracy = {accuracy:.2%}"
        )

        if accuracy >= 0.96:
            break
