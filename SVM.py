def vocaburary_mapping():
    vocab_list = {}
    with open('vocab.txt', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
    vocab_list[row[1]] = int(row[0])
    return vocab_list 


def email_preprocess(email):
    # 读取指定邮件文本
    with open(email, 'r') as f:
    email_contents = f.read()
    vocab_list = vocaburary_mapping()
    word_indices = []
    # 邮件文本预处理
    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = re.split('[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%\n") + ']', email_contents)
    for token in tokens:
    token = re.sub('[^a-zA-Z0-9]', '', token)
    token = stemmer.stem(token.strip())
    if len(token) == 0:
    continue
    if token in vocab_list:
    word_indices.append(vocab_list[token])
    # 返回邮件文本词汇与词典中词汇的对应关系，以及得到的预处理文本
    return word_indices, ' '.join(tokens)


word_indices, processed_contents = email_preprocess('emailSample1.txt')
print(word_indices)
print(processed_contents)


word_indices, processed_contents = email_preprocess('emailSample1.txt')
print(word_indices)
print(processed_contents)

word_indices, processed_contents = email_preprocess('emailSample1.txt')
print(word_indices)
print(processed_contents)

#加载训练集
mat = scipy.io.loadmat("spamTrain.mat")
X, y = mat['X'], mat['y']
#训练 SVM
linear_svm = svm.SVC(C=0.1, kernel='linear')
linear_svm.fit(X, y.ravel())
# 预测并计算训练集正确率
predictions = linear_svm.predict(X)
predictions = predictions.reshape(np.shape(predictions)[0], 1)
print('{}%'.format((predictions == y).mean() * 100.0))
# 加载测试集
mat = scipy.io.loadmat("spamTest.mat")
X_test, y_test = mat['Xtest'], mat['ytest']
# 预测并计算测试集正确率
predictions = linear_svm.predict(X_test)
predictions = predictions.reshape(np.shape(predictions)[0], 1)
print('{}%'.format((predictions == y_test).mean() * 100.0))


vocab_list = vocaburary_mapping()
reversed_vocab_list = dict((v, k) for (k, v) in vocab_list.items())
sorted_indices = np.argsort(linear_svm.coef_, axis=None)
for i in sorted_indices[0:15]:
print(reversed_vocab_list[i])
